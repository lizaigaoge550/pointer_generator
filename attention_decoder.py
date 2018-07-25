import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.ops.distributions import bernoulli




def linear(args, output_size, bias, bias_start=0.0, scope=None):
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear", reuse=tf.AUTO_REUSE):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term





def attention(decoder_state, attention_vec_size, encoder_features, enc_padding_mask, v,
              batch_size, enc_states, attn_size,
              coverage=None, w_c=None):
    with variable_scope.variable_scope("Attention"):
        decoder_features = linear(decoder_state, attention_vec_size, True)  # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)


        if coverage is not None:
            coverage_features = nn_ops.conv2d(coverage, w_c,[1,1,1,1],"SAME")
            e_not_masked = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features + coverage_features),[2, 3])
            masked_e = nn_ops.softmax(e_not_masked) * enc_padding_mask  # (batch_size, max_enc_steps)
            masked_sums = tf.reduce_sum(masked_e, axis=1)  # shape (batch_size)
            masked_e = masked_e / tf.reshape(masked_sums, [-1, 1])
            attn_dist = masked_e
            masked_attn_sums = tf.reduce_sum(attn_dist, axis=1)
            attn_dist = attn_dist / tf.reshape(masked_attn_sums, [-1, 1])  # re-normalize
            coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])

        else:
            e_not_masked = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),[2, 3])  # calculate e, (batch_size, max_enc_steps)
            masked_e = nn_ops.softmax(e_not_masked) * enc_padding_mask  # (batch_size, max_enc_steps)
            masked_sums = tf.reduce_sum(masked_e, axis=1)  # shape (batch_size)
            masked_e = masked_e / tf.reshape(masked_sums, [-1, 1])
            attn_dist = masked_e
            masked_attn_sums = tf.reduce_sum(attn_dist, axis=1)
            attn_dist = attn_dist / tf.reshape(masked_attn_sums, [-1, 1])  # re-normalize

        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * enc_states,
                                             [1, 2])  # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

    return context_vector, attn_dist, coverage, masked_e


def _calc_final_dist(_hps, v_size, _max_art_oovs, _enc_batch_extend_vocab, p_gen, vocab_dist, attn_dist):
  with tf.variable_scope('final_distribution'):
    vocab_dist = p_gen * vocab_dist
    attn_dist = (1-p_gen) * attn_dist

    extended_vsize = v_size + _max_art_oovs # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((_hps.batch_size, _max_art_oovs))
    vocab_dists_extended = tf.concat(axis=1, values=[vocab_dist, extra_zeros]) # list length max_dec_steps of shape (batch_size, extended_vsize)

    batch_nums = tf.range(0, limit=_hps.batch_size) # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1] # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
    indices = tf.stack( (batch_nums, _enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
    shape = [_hps.batch_size, extended_vsize]
    attn_dists_projected = tf.scatter_nd(indices, attn_dist, shape) # list length max_dec_steps (batch_size, extended_vsize)

    final_dist = vocab_dists_extended + attn_dists_projected
    final_dist +=1e-15 # for cases where we have zero in the final dist, especially for oov words
    dist_sums = tf.reduce_sum(final_dist, axis=1)
    final_dist = final_dist / tf.reshape(dist_sums, [-1, 1]) # re-normalize
  return final_dist



def attention_decoder(_hps,  v_size,  _max_art_oovs,  _enc_batch_extend_vocab,  emb_dec_inputs,  _dec_in_state,
  _enc_states, enc_padding_mask, dec_padding_mask, cell,
  initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None, embedding=None, schedule_sampling=None,
                      prev_decoder_outputs=[]):

    def intra_decoder_attention(decoder_state, outputs):
        try:
            len_dec_states = outputs.get_shape()[0].value
        except:
            len_dec_states = 0
        if len_dec_states == None:
            len_dec_states = 0
        print('len_dec_states : {0}'.format(len_dec_states))
        attention_dec_vec_size = decoder_state.c.get_shape()[1].value
        _decoder_states = tf.expand_dims(tf.reshape(outputs, [batch_size, -1, attention_dec_vec_size]), axis=2)
        _prev_decoder_features = nn_ops.conv2d(_decoder_states, W_h_d, [1,1,1,1], "SAME") #[batch_size, len(dec), 1 , dec_hidden]
        with variable_scope.variable_scope("DecoderAttention"):
            if len_dec_states > 0:
                decoder_features = linear(decoder_state, attention_dec_vec_size, True) #[batch_size, dec_hidden]
                decoder_features = tf.expand_dims(tf.reshape(decoder_features, [batch_size, -1, attention_dec_vec_size]),axis=1) #[batch_size, l ,1, dec_hidden]
                e_not_masked = math_ops.reduce_sum(v_d * math_ops.tanh(_prev_decoder_features + decoder_features), [2,3]) #[batch_size, len(dec)]
                masked_e = nn_ops.softmax(e_not_masked) * dec_padding_mask[:, :len_dec_states]
                masked_sums = tf.reshape(tf.reduce_sum(masked_e, axis=1), [-1,1])  # (batch_size,1), # if it's zero due to masking we set it to a small value
                decoder_attn_dist = masked_e / masked_sums  # (batch_size,len(decoder_states))
                context_decoder_vector = math_ops.reduce_sum(
                    array_ops.reshape(decoder_attn_dist, [batch_size, -1, 1, 1]) * _decoder_states,
                    [1, 2])  # (batch_size, attn_size)
                context_decoder_vector = array_ops.reshape(context_decoder_vector,
                                                       [-1, attention_dec_vec_size])  # (batch_size, attn_size)
            else:
                return array_ops.zeros([batch_size, attention_dec_vec_size])
            return context_decoder_vector

    attn_dists = []
    p_gens = []
    outputs = []
    vocab_scores = []
    vocab_dists = []
    #samples = []
    greedy_search_samples = []
    final_dists = []

    with variable_scope.variable_scope("attention_decoder") as scope:
        batch_size = _enc_states.get_shape()[0].value
        attn_size = _enc_states.get_shape()[2].value
        emb_size = emb_dec_inputs[0].get_shape()[1].value
        decoder_attn_size = _dec_in_state.c.get_shape()[1].value
        tf.logging.info("batch_size %i, attn_size: %i, emb_size: %i", batch_size, attn_size, emb_size)
        # Reshape _enc_states (need to insert a dim)
        _enc_states = tf.expand_dims(_enc_states, axis=2)

        attention_vec_size = attn_size
        W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        v = variable_scope.get_variable("v", [attention_vec_size])

        w_c = None
        if use_coverage:
            with variable_scope.variable_scope("coverage"):
                w_c = variable_scope.get_variable("w_c",[1, 1, 1, attention_vec_size])
        if prev_coverage is not None:
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage,2),3)

        if _hps.intradecoder:
            W_h_d = variable_scope.get_variable("W_h_d", [1, 1, decoder_attn_size, decoder_attn_size])
            v_d = variable_scope.get_variable("v_d", [decoder_attn_size])

        encoder_features = nn_ops.conv2d(_enc_states, W_h, [1, 1, 1, 1], "SAME")

        state = _dec_in_state
        coverage = prev_coverage
        context_vector = array_ops.zeros([batch_size, attn_size])
        context_decoder_vector = array_ops.zeros([batch_size, decoder_attn_size])
        context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.

        if initial_state_attention:
            context_vector, _, coverage, _ = attention(_dec_in_state, attention_vec_size, encoder_features, enc_padding_mask, v, batch_size,
                                                                       _enc_states, attn_size , coverage, w_c)
            if _hps.intradecoder:
                context_decoder_vector = intra_decoder_attention(_dec_in_state,
                                                                    tf.stack(prev_decoder_outputs, axis=0))

        for i, inp in enumerate(emb_dec_inputs):
            tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(emb_dec_inputs))
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            emb_dim = inp.get_shape().with_rank(2)[1]
            if emb_dim.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            if _hps.scheduled_sampling:
                if i > 0 and _hps.mode in ['train','eval']:
                    inp = scheduled_sampling(_hps, schedule_sampling, final_dist, embedding, inp)



            x = linear([inp] + [context_vector], emb_dim, True)
            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):  # you need this because you've already run the initial attention(...) call
                    #decoder_state, attention_vec_size, encoder_features, enc_padding_mask, v, batch_size, enc_states, attn_size
                    context_vector, attn_dist, _, masked_e = attention(state, attention_vec_size, encoder_features, enc_padding_mask, v, batch_size,
                                                                       _enc_states, attn_size , coverage, w_c)  # don't allow coverage to update
                    if _hps.intradecoder:
                        context_decoder_vector = intra_decoder_attention(state, tf.stack(prev_decoder_outputs, axis=0))
            else:
                context_vector, attn_dist, coverage, masked_e = attention(state, attention_vec_size, encoder_features, enc_padding_mask,
                                                                          v, batch_size, _enc_states, attn_size , coverage, w_c)
                if _hps.intradecoder:
                    context_decoder_vector = intra_decoder_attention(state, tf.stack(outputs,axis=0))

            attn_dists.append(attn_dist)

            if _hps.intradecoder:
                #print("context_vector : {0}".format(context_vector.get_shape().as_list()))
                #print("context_decoder_vector : {0}".format(context_decoder_vector.get_shape().as_list()))
                with tf.variable_scope('combine_context'):
                    context_vector = linear([context_vector] + [context_decoder_vector], attention_vec_size, False)


            # Calculate p_gen
            if pointer_gen:
                with tf.variable_scope('calculate_pgen'):
                    p_gen = linear([context_vector, state.c, state.h, x], 1, True)  # Tensor shape (batch_size, 1)
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

            # Add the output projection to obtain the vocabulary distribution
            with tf.variable_scope('output_projection'):
                trunc_norm_init = tf.truncated_normal_initializer(stddev=_hps.trunc_norm_init_std)
                w_out = tf.get_variable('w', [_hps.dec_hidden_dim, v_size], dtype=tf.float32,
                                    initializer=trunc_norm_init)
                v_out = tf.get_variable('v', [v_size], dtype=tf.float32, initializer=trunc_norm_init)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                score = tf.nn.xw_plus_b(output, w_out, v_out)
                vocab_scores.append(score)  # apply the linear layer
                vocab_dist = tf.nn.softmax(score)
                vocab_dists.append(vocab_dist)

            # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
            if _hps.pointer_gen:
                final_dist = _calc_final_dist(_hps, v_size, _max_art_oovs, _enc_batch_extend_vocab, p_gen, vocab_dist,
                                              attn_dist)

            final_dists.append(final_dist)


            #one_hot_k_samples = tf.distributions.Multinomial(total_count=1., probs=final_dist).sample(_hps.k)  # sample once according to https://arxiv.org/pdf/1705.04304.pdf, size (k,batch_size,extended_vsize)
            #k_argmax = tf.argmax(one_hot_k_samples, axis=2, output_type=tf.int32)  # (k, batch_size)
            #k_sample = tf.transpose(k_argmax)  # this will take the final_dist and sample from it for a total count of k (k samples), the result is of shape (batch_size,k)
            greedy_search_prob, greedy_search_sample = tf.nn.top_k(final_dist, k=1)  # (batch_size, k)
            greedy_search_samples.append(greedy_search_sample)
            #samples.append(k_sample)
            if coverage is not None:
                coverage = array_ops.reshape(coverage, [batch_size,-1])

    return outputs, state, attn_dists, p_gens, coverage, vocab_scores, final_dists, greedy_search_samples

def scheduled_sampling(hps, sampling_probability, output, embedding, inp):
    vocab_size = embedding.get_shape()[0].value
    with variable_scope.variable_scope("ScheduleEmbedding"):
        select_sampler = bernoulli.Bernoulli(probs=sampling_probability, dtype=tf.bool)
        select_sample = select_sampler.sample(sample_shape=hps.batch_size)
        sample_id_sampler = categorical.Categorical(probs=output)
        sample_ids = array_ops.where(select_sample, sample_id_sampler.sample(seed=123), gen_array_ops.fill([hps.batch_size],-1))
        where_sampling = math_ops.cast(array_ops.where(sample_ids > -1), tf.int32)
        where_not_sampling = math_ops.cast(array_ops.where(sample_ids <= -1), tf.int32)
        sample_ids_sampling = array_ops.gather_nd(sample_ids, where_sampling)
        cond = tf.less(sample_ids_sampling, vocab_size)
        sample_ids_sampling = tf.cast(cond, tf.int32) * sample_ids_sampling
        inputs_not_sampling = array_ops.gather_nd(inp, where_not_sampling)
        sampling_next_inputs = tf.nn.embedding_lookup(embedding, sample_ids_sampling)
        result1 = array_ops.scatter_nd(indices=where_sampling, updates=sampling_next_inputs, shape=array_ops.shape(inp))
        result2 = array_ops.scatter_nd(indices=where_not_sampling, updates=inputs_not_sampling, shape=array_ops.shape(inp))
        return result1 + result2
