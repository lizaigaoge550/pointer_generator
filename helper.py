import os
import tensorflow as tf
from rouge import rouge
import glob
def make_feed_dict(model, batch, just_enc=False):
    feed_dict = {}
    feed_dict[model._enc_batch] = batch.enc_batch
    feed_dict[model._enc_lens] = batch.enc_lens
    feed_dict[model._enc_padding_mask] = batch.enc_padding_mask
    feed_dict[model._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
    feed_dict[model._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
        feed_dict[model._dec_batch] = batch.dec_batch
        feed_dict[model._target_batch] = batch.target_batch
        feed_dict[model._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  #config = tf.ConfigProto(log_device_placement=True)
  config.gpu_options.allow_growth=True
  return config

def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s

def preprocess(article):
    article = article.split()
    new_article = []
    start = 0
    end = 50
    while end < len(article):
        new_article.append(" ".join(article[start:end]))
        end = start + 50
        start = end
    if end < len(article):
        new_article.append(" ".join(article[start:len(article)]))
    return new_article

def write_for_rouge_beam(reference_sents, decoded_words, article, ex_index, dec_dir, ref_dir, all_dir):

    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:  # there is text remaining that doesn't end in "."
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
        decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    #if not os.path.exists(dec_dir):os.mkdir(dec_dir)
    #if not os.path.exists(ref_dir): os.mkdir(ref_dir)
    if not os.path.exists(all_dir): os.mkdir(all_dir)
    #ref_file = os.path.join(ref_dir, "%06d_reference.txt" % ex_index)
    #decoded_file = os.path.join(dec_dir, "%06d_decoded.txt" % ex_index)
    all_file = os.path.join(all_dir, "%06d_decoded.txt" % ex_index)
    #with open(ref_file, "w") as f:
    #    for idx, sent in enumerate(reference_sents):
    #        f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    #with open(decoded_file, "w") as f:
    #    for idx, sent in enumerate(decoded_sents):
    #        f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")
    with open(all_file, "w") as f:
       f.write('article : \n')
        #articles = preprocess(article)
        #for article in articles:
       f.write(article + '\n')
       f.write('\ndecode :\n')
       for idx, sent in enumerate(decoded_sents):
           f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")
       f.write('\nref :\n')
       for idx, sent in enumerate(reference_sents):
           f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


def write_for_rouge_greedy(reference_sents, decoded_words, article, ex_index, dec_dir, ref_dir, all_dir):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    for i in range(len(decoded_words)):
        decoded_word = decoded_words[i]
        reference_sent = reference_sents[i]
        decoded_sents = []
        while len(decoded_word) > 0:
            try:
                fst_period_idx = decoded_word.index(".")
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_word)
            sent = decoded_word[:fst_period_idx + 1]  # sentence up to and including the period
            decoded_word = decoded_word[fst_period_idx + 1:]  # everything else
            decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sent = [make_html_safe(w) for w in decoded_sents]
        reference_sent = [make_html_safe(w) for w in reference_sent]

        # Write to file
        #if not os.path.exists(dec_dir):os.mkdir(dec_dir)
        #if not os.path.exists(ref_dir): os.mkdir(ref_dir)
        if not os.path.exists(all_dir): os.mkdir(all_dir)
        #ref_file = os.path.join(ref_dir, "%06d_reference.txt" % (ex_index+i))
        #decoded_file = os.path.join(dec_dir, "%06d_decoded.txt" % (ex_index+i))
        all_file = os.path.join(all_dir, "%06d_decoded.txt" % (ex_index+i))
        #with open(ref_file, "w") as f:
        #    for idx, sent in enumerate(reference_sent):
        #        f.write(sent) if idx == len(reference_sent) - 1 else f.write(sent + "\n")
        #with open(decoded_file, "w") as f:
        #    for idx, sent in enumerate(decoded_sent):
        #        f.write(sent) if idx == len(decoded_sent) - 1 else f.write(sent + "\n")
        with open(all_file, "w") as f:
             f.write('article : \n')
             articles = preprocess(article)
             for article in articles:
                 f.write(article + '\n')
             f.write('\ndecode :\n')
             for idx, sent in enumerate(decoded_sent):
                 f.write(sent) if idx == len(decoded_sent) - 1 else f.write(sent + "\n")
             f.write('\nref :\n')
             for idx, sent in enumerate(reference_sent):
                 f.write(sent) if idx == len(reference_sent) - 1 else f.write(sent + "\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


def reward_function(reference, summary, measure='rouge_l/f_score'):

    if 'rouge' in measure:
        return rouge([summary], [reference])[measure]

def remove_stop_index(decode_words, data):
    decode_word = []
    for item in decode_words:
        if data.STOP_DECODING in item:
            decode_word.append(item[:data.STOP_DECODING])
        else:
            decode_word.append(item)
    return decode_word


def reader_params(path):
    reader = tf.train.NewCheckpointReader(path)
    var_to_map = reader.get_variable_to_shape_map()
    dic = {}
    fw = open('pretain_params.txt','w')
    for v in var_to_map:
        fw.write(v+'\n')
    #    print(v)
        dic[v] = reader.get_tensor(v)
    return dic

def load_best_model(dir):
    model_set = set()
    for f in glob.glob(os.path.join(dir,'*')):
        f = f.split('/')[-1]
        if 'model' in f:
            f = f[:f.index('ckpt')-1]
            model_set.add(f)
    models = [a for a in list(model_set) if 'model' in a]
    best_file = sorted(models, key=lambda a:int(a.split('_')[1]), reverse=True)[0]
    return os.path.join(dir, best_file+'.ckpt')

def loading_variable(src_params, dst_params):
    p_l = []
    for s_p in src_params:
        p_l.append(s_p.assign(dst_params[s_p.name.split(':')[0]]))
    return p_l
