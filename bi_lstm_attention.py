from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
from  attention_decoder import linear

def attention(state, outputs, w , v, enc_padding_mask):
    batch_size = state.c.get_shape()[0].value
    len_dec_states = outputs.get_shape()[0].value
    assert len_dec_states > 0
    attention_vec_size = state.c.get_shape()[1].value
    _decoder_states = tf.expand_dims(tf.reshape(outputs, [batch_size, -1, attention_vec_size]), axis=2)
    _prev_decoder_features = tf.nn.conv2d(_decoder_states, w, [1, 1, 1, 1],
                                           "SAME")  # [batch_size, len(dec), 1 , dec_hidden]
    with tf.variable_scope("ecoderAttention"):

        decoder_features = linear(state, attention_vec_size, True)  # [batch_size, dec_hidden]
        decoder_features = tf.expand_dims(tf.reshape(decoder_features, [batch_size, -1, attention_vec_size]),
                                          axis=1)  # [batch_size, l ,1, dec_hidden]
        e_not_masked = tf.reduce_sum(v * tf.tanh(_prev_decoder_features + decoder_features),
                                           [2, 3])  # [batch_size, len(dec)]
        masked_e = tf.nn.softmax(e_not_masked) * enc_padding_mask[:, :len_dec_states]
        masked_sums = tf.reshape(tf.reduce_sum(masked_e, axis=1), [-1,
                                                                   1])  # (batch_size,1), # if it's zero due to masking we set it to a small value
        decoder_attn_dist = masked_e / masked_sums  # (batch_size,len(decoder_states))
        context_decoder_vector = tf.reduce_sum(
            array_ops.reshape(decoder_attn_dist, [batch_size, -1, 1, 1]) * _decoder_states,
            [1, 2])  # (batch_size, attn_size)
        context_decoder_vector = array_ops.reshape(context_decoder_vector,
                                                   [-1, attention_vec_size])  # (batch_size, attn_size)

        return context_decoder_vector


def initial_state(shape, type):
    c = tf.zeros(shape=shape, dtype=type)
    h = tf.zeros(shape=shape, dtype=type)
    return tf.nn.rnn_cell.LSTMStateTuple(c,h)



def bi_lstm_attnetion(inputs, sequence_length, cell_fw, cell_bw, W_forward, v_forward, W_backward, v_backward,
                      enc_padding_mask, scope=None, time_major=False):

    rnn_cell_impl.assert_like_rnncell("cell_fw", cell_fw)
    rnn_cell_impl.assert_like_rnncell("cell_bw", cell_bw)

    time_step = inputs.get_shape()[1].value
    outputs_fw = []
    outputs_bw = []
    batch_size = inputs.get_shape()[0].value
    emb_dim = inputs.get_shape()[-1].value
    state_fw = initial_state(shape=[batch_size, cell_bw._num_units], type=inputs.dtype)
    state_bw = initial_state(shape=[batch_size, cell_bw._num_units], type=inputs.dtype)
    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        context_vector = tf.zeros(shape=[batch_size, cell_bw._num_units], dtype=inputs.dtype)
        with vs.variable_scope("fw") as fw_scope:
            for i in range(time_step):
                print('encoder attention forward : {0}...'.format(i))
                input = inputs[:,i,:] #[batch,embedding_size]
                x = linear([input] + [context_vector], emb_dim, False)
                cell_output, state_fw = cell_fw(x, state_fw)
                if i > 1:
                    tf.get_variable_scope().reuse_variables()
                if i > 0:
                    context_vector = attention(state_fw, tf.stack(outputs_fw,axis=0), W_forward, v_forward, enc_padding_mask)
                outputs_fw.append(cell_output)
            outputs_state_fw = state_fw



        # Backward direction
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1

        def _reverse(input_, seq_lengths, seq_dim, batch_dim):
            if seq_lengths is not None:
                return array_ops.reverse_sequence(
                    input=input_, seq_lengths=seq_lengths,
                    seq_dim=seq_dim, batch_dim=batch_dim)
            else:
                return array_ops.reverse(input_, axis=[seq_dim])

        with vs.variable_scope("bw") as bw_scope:
            context_vector = tf.zeros(shape=[batch_size, cell_bw._num_units], dtype=inputs.dtype)
            for i in range(time_step-1,-1,-1):
                print('encoder attention backward : {0}...'.format(i))
                input = inputs[:, i, :]  # [batch,embedding_size]
                x = linear([input] + [context_vector], emb_dim, False)
                cell_output, state_bw = cell_fw(x, state_bw)
                if i < time_step - 2:
                    tf.get_variable_scope().reuse_variables()
                if i < time_step - 1:
                    context_vector = attention(state_fw, tf.stack(outputs_fw, axis=0), W_backward, v_backward,
                                               enc_padding_mask[::-1])
                outputs_bw.append(cell_output)
            outputs_state_bw = state_bw

    tmp = tf.stack(outputs_bw,axis=1)
    outputs_fw = tf.stack(outputs_fw,axis=1)
    outputs_bw = _reverse(
        tmp, seq_lengths=sequence_length,
        seq_dim=time_dim, batch_dim=batch_dim)

    outputs = (outputs_fw, outputs_bw)
    output_states = (outputs_state_fw, outputs_state_bw)

    return (outputs, output_states)
