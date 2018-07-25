# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""

from queue import Queue
from random import shuffle
from random import seed
seed(123)
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data


class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, article, abstract_sentences, vocab, hps):
    """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

    Args:
      article: source text; a string. each token is separated by a single space.
      abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
      vocab: Vocabulary object
      hps: hyperparameters
    """
    self.hps = hps
    self.flag = False
    # Get ids of special tokens
    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)

    # Process the article
    article_words = article.split()
    if len(article_words) > hps.max_enc_steps:
      article_words = article_words[:hps.max_enc_steps]
    self.enc_len = len(article_words) # store the length after truncation but before padding
    self.enc_input = [vocab.word2id(w) for w in article_words] # list of word ids; OOVs are represented by the id for UNK token
    if len(self.enc_input) == 0 or sum(self.enc_input) == 0:
        self.flag = True
    # Process the abstract
    abstract = ' '.join(abstract_sentences) # string
    abstract_words = abstract.split() # list of strings
    abs_ids = [vocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token

    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)

    # If using pointer-generator mode, we need to store some extra info
    if hps.pointer_gen:
      # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
      self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

      # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
      abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

      # Overwrite decoder target sequence so it uses the temp article OOV ids
      _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

    # Store the original strings
    self.original_article = article
    self.original_abstract = abstract
    self.original_abstract_sents = abstract_sentences


  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer

    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target


  def pad_decoder_inp_targ(self, max_len, pad_id):
    """Pad decoder input and target sequences with pad_id up to max_len."""
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)


  def pad_encoder_input(self, max_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
    if self.hps.pointer_gen:
      while len(self.enc_input_extend_vocab) < max_len:
        self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
    self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings

  def init_encoder_seq(self, example_list, hps):
    """Initializes the following:
        self.enc_batch:
          numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
        self.enc_lens:
          numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
        self.enc_padding_mask:
          numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

      If hps.pointer_gen, additionally initializes the following:
        self.max_art_oovs:
          maximum number of in-article OOVs in the batch
        self.art_oovs:
          list of list of in-article OOVs (strings), for each example in the batch
        self.enc_batch_extend_vocab:
          Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
    """
    # Determine the maximum length of the encoder input sequence in this batch
    #max_enc_seq_len = max([ex.enc_len for ex in example_list])
    max_enc_seq_len = hps.max_enc_steps
    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
        self.enc_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    if hps.pointer_gen:
      # Determine the max number of in-article OOVs in this batch
      self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
      # Store the in-article OOVs themselves
      self.art_oovs = [ex.article_oovs for ex in example_list]
      # Store the version of the enc_batch that uses the article OOV ids
      self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  def init_decoder_seq(self, example_list, hps):
    """Initializes the following:
        self.dec_batch:
          numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
        self.target_batch:
          numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
        self.dec_padding_mask:
          numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
        """
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
    self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      for j in range(ex.dec_len):
        self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    """Store the original article and abstract strings in the Batch object"""
    self.original_articles = [ex.original_article for ex in example_list] # list of lists
    self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
    self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists


class Batcher(object):
  """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, hps, single_pass, decode_after):
    """Initialize the batcher. Start threads that process the data into batches.

    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary object
      hps: hyperparameters
      single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
    """
    self._data_path = data_path
    self._vocab = vocab
    self._hps = hps
    self._single_pass = single_pass
    self._decode_after = decode_after

    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = []
    self._example_queue = []



  def fill_example_queue(self):
    """Reads data from file and processes into Examples which are then placed into the example queue."""
    input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))
    self._example_queue = []
    for item in input_gen:
        article, abstract = str(item[0]), str(item[1])
        abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)]
        example = Example(article, abstract_sentences, self._vocab, self._hps)
        if example.flag == False:
            self._example_queue.append(example)


  def fill_batch_queue(self, is_training=True):
    if len(self._example_queue) == 0:
        self.fill_example_queue()
    print('****************data_len:{0}****************'.format(len(self._example_queue)))
    if len(self._example_queue) == 0:
        tf.logging.error('batcher: data len is 0')
        exit(1)
    data_len = len(self._example_queue)
    if self._hps.beam == False:
        num_batch_per_epoch = int(data_len/self._hps.batch_size) + 1
        if is_training:
            np.random.shuffle(self._example_queue)
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num  * self._hps.batch_size
            if(batch_num + 1) * self._hps.batch_size <= data_len:
                end_index = (batch_num + 1) * self._hps.batch_size
                yield Batch(self._example_queue[start_index:end_index], self._hps, self._vocab)
    else:
        for i in range(len(self._example_queue)):
            b = [self._example_queue[i] for _ in range(self._hps.batch_size)]
            yield Batch(b,self._hps,self._vocab)
  def text_generator(self, example_generator):
    text = []
    for e in example_generator:
        article_text = e.features.feature['article'].bytes_list.value[0].decode('utf-8')
        abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode('utf-8')
        text.append((article_text, abstract_text))
    return text
