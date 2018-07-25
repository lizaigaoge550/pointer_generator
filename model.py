import tensorflow as tf
from attention_decoder import attention_decoder
import numpy as np
from helper import make_feed_dict
from bi_lstm_attention import bi_lstm_attnetion
import numpy as np
FLAGS = tf.app.flags.FLAGS

def _mask_and_avg(values, padding_mask):
  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)] # list of k
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average

def _coverage_loss(attn_dists, padding_mask):
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss


class PointerNet(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab
        print("coverage ... ", self._hps.coverage)
    def build_graph(self):
        tf.logging.info('Building graph...')
        gpu_num = self._hps.gpu_num.split(',')
        with tf.device('/gpu:{0}'.format(gpu_num[0])):
            self._add_placeholders()
            self._add_seq2seq()
            if self._hps.mode in ['train','eval']:
                self._add_shared_loss_op()
                self.loss_function()

    def _add_placeholders(self):
        hps = self._hps
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_enc_steps], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
        self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
        self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps],
                                                name='dec_padding_mask')



        if hps.mode == "decode":
            if hps.coverage:
                self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')
            if hps.intradecoder:
                self.prev_decoder_outputs = tf.placeholder(tf.float32, [None, hps.batch_size, hps.dec_hidden_dim],
                                                           name='prev_decoder_outputs')

    def selectNet(self, enc_outputs):
        s = enc_outputs[:, -1, :]
        new_enc_outputs = []
        with tf.variable_scope('selectNet'):
            self.Ws = tf.get_variable('Ws', [2 * self._hps.enc_hidden_dim, 2 * self._hps.enc_hidden_dim],
                                      initializer=self.trunc_norm_init)
            self.Us = tf.get_variable('Us', [2 * self._hps.enc_hidden_dim, 2 * self._hps.enc_hidden_dim],
                                      initializer=self.trunc_norm_init)
            self.bs = tf.get_variable('bs', [2 * self._hps.enc_hidden_dim], initializer=tf.zeros_initializer)
            for i in range(self._hps.max_enc_steps):
                new_enc_outputs.append(
                    tf.sigmoid(tf.matmul(enc_outputs[:, i, :], self.Ws) + tf.matmul(s, self.Us) + self.bs))
        return tf.transpose(tf.squeeze(new_enc_outputs), [1, 0, 2])





    def run_encoder(self, sess, batch):
        feed_dict = make_feed_dict(self, batch, just_enc=True)  # feed the batch into the placeholders
        (enc_states, dec_in_state) = sess.run([self._enc_states, self._dec_in_state],
                                                           feed_dict)  # run the encoder
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        return enc_states, dec_in_state


    def _add_seq2seq(self):
        hps = self._hps
        vsize = self._vocab.size()  # size of the vocabulary

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
            self.embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,
                                             initializer=self.trunc_norm_init)
            emb_enc_inputs = tf.nn.embedding_lookup(self.embedding,
                                                    self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
            emb_dec_inputs = [tf.nn.embedding_lookup(self.embedding, x) for x in tf.unstack(self._dec_batch, axis=1)]
            with tf.variable_scope('encoder'):
                enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
                self._enc_states = enc_outputs
                self._dec_in_state = self._reduce_states(fw_st, bw_st)
            with tf.variable_scope('decoder'):
                #outputs, state, attn_dists, p_gens, coverage, vocab_scores, final_dists, greedy_search_samples
                self.decoder_outputs, self.state, self.attn_dists, self.p_gens, self.coverage, self.vocab_scores, self.final_dists, \
                self.greedy_search_samples = \
                    self._add_decoder(emb_dec_inputs)

            # TODO
            #self.sampled_sentences = tf.unstack(tf.stack(self.samples, axis=2))
            self.greedy_search_sentences = tf.unstack(tf.stack(self.greedy_search_samples, axis=2))
        if hps.mode == "decode":
            # We run decode beam search mode one decoder step at a time
            assert len(
                self.final_dists) == 1  # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            final_dists = self.final_dists[0]
            if FLAGS.beam == True:
                topk_probs, self._topk_ids = tf.nn.top_k(final_dists,
                                                     hps.batch_size * 2)  # take the k largest probs. note batch_size=beam_size in decode mode
            else:
                topk_probs, self._topk_ids = tf.nn.top_k(final_dists,1)
            self._topk_log_probs = tf.log(topk_probs)

    def _add_encoder(self, emb_enc_inputs, seq_len):
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.enc_hidden_dim, initializer=tf.zeros_initializer,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.enc_hidden_dim, initializer=tf.zeros_initializer,
                                              state_is_tuple=True)

            if self._hps.encoder_attention:
                if len(self._hps.gpu_num.split(',')) > 1:
                    with tf.device('/gpu:{0}'.format(self._hps.gpu_num.split(",")[1])):
                        W_forward = tf.get_variable(name="encoder_attn_w_forward", shape=[1,1,self._hps.enc_hidden_dim, self._hps.enc_hidden_dim],initializer=self.trunc_norm_init)
                        v_forward = tf.get_variable(name='encoder_attn_v_forward', shape=[self._hps.enc_hidden_dim], initializer=tf.zeros_initializer)
                        W_backward = tf.get_variable(name="encoder_attn_w_backward", shape=[1, 1, self._hps.enc_hidden_dim, self._hps.enc_hidden_dim],
                                                initializer=self.trunc_norm_init)
                        v_backward = tf.get_variable(name='encoder_attn_v_backward', shape=[self._hps.enc_hidden_dim],
                                                initializer=tf.zeros_initializer)
                        (encoder_outputs, (fw_st, bw_st)) = bi_lstm_attnetion(emb_enc_inputs, seq_len, cell_fw, cell_bw,W_forward,
                                                                              v_forward,W_backward,v_backward,self._enc_padding_mask)

                else:
                    W_forward = tf.get_variable(name="encoder_attn_w_forward",
                                                shape=[1, 1, self._hps.enc_hidden_dim, self._hps.enc_hidden_dim],
                                                initializer=self.trunc_norm_init)
                    v_forward = tf.get_variable(name='encoder_attn_v_forward', shape=[self._hps.enc_hidden_dim],
                                                initializer=tf.zeros_initializer)
                    W_backward = tf.get_variable(name="encoder_attn_w_backward",
                                                 shape=[1, 1, self._hps.enc_hidden_dim, self._hps.enc_hidden_dim],
                                                 initializer=self.trunc_norm_init)
                    v_backward = tf.get_variable(name='encoder_attn_v_backward', shape=[self._hps.enc_hidden_dim],
                                                 initializer=tf.zeros_initializer)
                    (encoder_outputs, (fw_st, bw_st)) = bi_lstm_attnetion(emb_enc_inputs, seq_len, cell_fw, cell_bw,
                                                                          W_forward,
                                                                          v_forward, W_backward, v_backward,
                                                                          self._enc_padding_mask)
            else:
                (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, emb_enc_inputs,
                                                                            dtype=tf.float32,
                                                                            sequence_length=seq_len,
                                                                            swap_memory=True)

            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
            if self._hps.select:
                encoder_outputs = self.selectNet(encoder_outputs)
            return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        enc_hidden_dim = self._hps.enc_hidden_dim
        dec_hidden_dim = self._hps.dec_hidden_dim

        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [enc_hidden_dim * 2, dec_hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [enc_hidden_dim * 2, dec_hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [dec_hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [dec_hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            with tf.device("/gpu:{0}".format(0)):
                # Apply linear layer
                old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
                old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
                new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
                new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state


#_hps,  v_size,  _max_art_oovs,  _enc_batch_extend_vocab,  emb_dec_inputs,  _dec_in_state, _enc_states, enc_padding_mask, cell,
  #initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None, prev_encoder_es=[]
    def _add_decoder(self, emb_dec_inputs):
        hps = self._hps
        cell = tf.contrib.rnn.LSTMCell(hps.dec_hidden_dim, state_is_tuple=True, initializer=tf.zeros_initializer)
        prev_coverage = self.prev_coverage if (hps.mode == "decode" and hps.coverage) else None
        prev_decoder_outputs = self.prev_decoder_outputs if (hps.intradecoder and hps.mode == "decode") else tf.stack([], axis=0)
        return attention_decoder(hps,
                             self._vocab.size(),
                             self._max_art_oovs,
                             self._enc_batch_extend_vocab,
                             emb_dec_inputs,
                             self._dec_in_state,
                             self._enc_states,
                             self._enc_padding_mask,
                             self._dec_padding_mask,
                             cell,
                             initial_state_attention=(hps.mode == "decode"),
                             pointer_gen=hps.pointer_gen,
                             use_coverage=hps.coverage,
                             prev_coverage=prev_coverage,
                             prev_decoder_outputs=prev_decoder_outputs
                             )

    def _add_shared_loss_op(self):
        # Calculate the loss
        with tf.variable_scope('shared_loss'):
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            #### added by yaserkl@vt.edu: we just calculate these to monitor pgen_loss throughout time
            loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
            batch_nums = tf.range(0, limit=self._hps.batch_size)  # shape (batch_size)
            for dec_step, dist in enumerate(self.final_dists):
                targets = self._target_batch[:, dec_step]  # The indices of the target words. shape (batch_size)
                indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
                losses = -tf.log(gold_probs)
                loss_per_step.append(losses)
            self._pgen_loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

            if self._hps.coverage:
                with tf.variable_scope('coverage_loss'):
                    print('coverage.....')
                    self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)

    def loss_function(self):
        self.loss = self._pgen_loss
        if self._hps.coverage:
            self.loss = self._pgen_loss + self._hps.cov_loss_wt * self._coverage_loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
        optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),name='train_step')

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage, prev_decoder_outputs=None):
        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
            self._dec_padding_mask: np.ones((beam_size, 1), dtype=np.float32)
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self.state,
            "attn_dists": self.attn_dists,
            "final_dists": self.final_dists
        }

        if FLAGS.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        if FLAGS.intradecoder:
            to_return['output'] = self.decoder_outputs
            feed[self.prev_decoder_outputs] = prev_decoder_outputs

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(beam_size)]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()
        final_dists = results['final_dists'][0].tolist()

        if FLAGS.pointer_gen:
            assert len(results['p_gens']) == 1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        if FLAGS.intradecoder:
            output = results['output'][0]
            #print('output size:{0}'.format(np.shape(output)))
        else:
            output = None
        temporal_e = None

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, final_dists, p_gens, new_coverage, output, temporal_e


