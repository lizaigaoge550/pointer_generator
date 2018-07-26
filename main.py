import time
import os
import tensorflow as tf
import numpy as np
from data import Vocab
from training import run_rl_training_gamma
from batcher import Batcher
from model import PointerNet
from helper import get_config,write_for_rouge_beam,write_for_rouge_greedy
from helper import make_feed_dict,load_best_model
import beam_search
import data
from rl_model import RLNet
from helper import reward_function
from helper import reader_params,remove_stop_index
import tqdm

FLAGS = tf.flags.FLAGS

# Where to find data
tf.flags.DEFINE_string('data_path', '', 'data_path')
tf.flags.DEFINE_string('val_data_path', '', 'val_data_path')
tf.flags.DEFINE_string('checkpoint','','checkpoint')
tf.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.flags.DEFINE_integer('decode_after', 0, 'skip already decoded docs')
tf.flags.DEFINE_integer('epoch', 10, 'epoch')
tf.flags.DEFINE_integer('patient', 5, 'patient')
tf.flags.DEFINE_integer('eval_step', 2000, 'eval_step')

# Where to save output
tf.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.flags.DEFINE_integer('enc_hidden_dim', 256, 'dimension of RNN hidden states')
tf.flags.DEFINE_integer('dec_hidden_dim', 256, 'dimension of RNN hidden states')
tf.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.flags.DEFINE_integer('batch_size', 2, 'minibatch size')
tf.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.flags.DEFINE_integer('max_iter', 55000, 'max number of iterations')
tf.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.flags.DEFINE_float('gamma', 0, 'gamma')
tf.flags.DEFINE_float('attn_gamma', 0, 'gamma')
tf.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.flags.DEFINE_float('threshold', 0.00001, 'for gradient clipping')
tf.flags.DEFINE_string('embedding', None, 'path to the pre-trained embedding file')
tf.flags.DEFINE_string('gpu_num', '', 'which gpu to use to train the model')

# Pointer-generator or baseline model
tf.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
tf.flags.DEFINE_boolean('rl', False, 'If True, use pointer-generator model. If False, use baseline model.')
tf.flags.DEFINE_boolean('encoder_attention', False, 'If True, use pointer-generator model. If False, use baseline model.')
tf.flags.DEFINE_boolean('select', False, 'If True, use pointer-generator model. If False, use baseline model.')
tf.flags.DEFINE_string('restore_path', '', 'restore_path.')
tf.flags.DEFINE_string('dec_path', '', 'dec_path.')
tf.flags.DEFINE_string('ref_path', '', 'ref_path.')
tf.flags.DEFINE_string('all_path', '', 'all_path.')
tf.flags.DEFINE_boolean('beam', False , 'all_path.')
tf.flags.DEFINE_string('restore_rl_path', '', 'restore_rl_path')
# Pointer-generator with Self-Critic policy gradient: https://arxiv.org/pdf/1705.04304.pdf


# Coverage hyperparameters
tf.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.flags.DEFINE_boolean('intradecoder', False, 'intradecoder.')
tf.flags.DEFINE_boolean('scheduled_sampling', False, 'scheduled_sampling')
tf.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu_num
vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)


def loading_variable(src_params, dst_params):
    p_l = []
    for s_p in src_params:
        p_l.append(s_p.assign(dst_params[s_p.name.split(':')[0]]))
    return p_l

def run_eval(model, batcher, sess):
    print('eval start ......')
    loss = 0
    batches = batcher.fill_batch_queue(is_training=False)
    for batch in batches:
        feed_dict = make_feed_dict(model, batch)
        eloss = sess.run(model.loss, feed_dict)
        loss += eloss
    return loss

def run_rl_eval(model, batcher, sess, eta):
    loss = 0
    batches = batcher.fill_batch_queue(is_training=False)
    for batch in batches:
        feed_dict = make_feed_dict(model, batch)
        feed_dict[model._eta] = eta
        if FLAGS.coverage:
            eloss = sess.run(model._pgen_loss + FLAGS.cov_loss_wt*model._coverage_loss, feed_dict)
        else:
            eloss = sess.run(model._pgen_loss, feed_dict)
        loss += eloss
    return loss

def run_training():
    print('batch size', FLAGS.batch_size)
    summarizationModel = PointerNet(FLAGS, vocab)
    summarizationModel.build_graph()
    batcher = Batcher(FLAGS.data_path, vocab, FLAGS, single_pass=FLAGS.single_pass, decode_after=FLAGS.decode_after)
    val_batcher = Batcher(FLAGS.val_data_path, vocab, FLAGS, single_pass=FLAGS.single_pass, decode_after=FLAGS.decode_after)
    sess = tf.Session(config=get_config())
    sess.run(tf.global_variables_initializer())

    eval_loss = float('inf')
    saver = tf.train.Saver(max_to_keep=10)
    if FLAGS.restore_path:
        print('loading params...')
        saver.restore(sess, FLAGS.restore_path)
    epoch = FLAGS.epoch
    step = 0
    patient = FLAGS.patient
    while epoch > 0:
        batches = batcher.fill_batch_queue()
        print('load batch...')
        for batch in batches:
            print('start training...')
            step += 1
            feed_dict = make_feed_dict(summarizationModel, batch)
            loss, _ = sess.run([summarizationModel.loss, summarizationModel.train_op], feed_dict)
            print("epoch : {0}, step : {1}, loss : {2}".format(abs(epoch-FLAGS.epoch), step, loss))
            if step % FLAGS.eval_step == 0:
                eval_ = run_eval(summarizationModel, val_batcher, sess)
                if eval_ < eval_loss:
                    if not os.path.exists(FLAGS.checkpoint):os.mkdir(FLAGS.checkpoint)
                    saver.save(sess, save_path=os.path.join(FLAGS.checkpoint,'model_{0}_{1}.ckpt'.format(step,eval_)))
                    eval_loss = eval_
                    patient = FLAGS.patient
                print('eval loss : {0}'.format(eval_loss))
                if patient < 0:
                    break

                if eval_ - eval_loss > FLAGS.threshold:
                    patient -= 1

def run_rl_training():

    summarizationModel = RLNet(FLAGS, vocab)
    summarizationModel.build_graph()
    batcher = Batcher(FLAGS.data_path, vocab, FLAGS, single_pass=FLAGS.single_pass, decode_after=FLAGS.decode_after)
    val_batcher = Batcher(FLAGS.val_data_path, vocab, FLAGS, single_pass=FLAGS.single_pass,decode_after=FLAGS.decode_after)
    sess = tf.Session(config=get_config())
    saver = tf.train.Saver(max_to_keep=100)
    if FLAGS.restore_rl_path:
        saver.restore(sess, FLAGS.restore_rl_path)
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(loading_variable([v for v in tf.trainable_variables()], reader_params(load_best_model(FLAGS.restore_path))))
    print('loading params...')
    epoch = FLAGS.epoch
    step = 0
    eval_loss = float('inf')
    while epoch > 0:
        batches = batcher.fill_batch_queue()
        for batch in batches:
            step += 1
            sampled_sentence_r_values = []
            greedy_sentence_r_values = []
            feed_dict = make_feed_dict(summarizationModel,batch)

            to_return = {
                'sampled_sentences': summarizationModel.sampled_sentences,
                'greedy_search_sentences': summarizationModel.greedy_search_sentences
                }
            ret_dict = sess.run(to_return, feed_dict)
            # calculate reward
            for sampled_sentence, greedy_search_sentence, target_sentence in zip(ret_dict['sampled_sentences'], ret_dict['greedy_search_sentences'],
                                                                         batch.target_batch):
                assert len(sampled_sentence[0]) == len(target_sentence) == len(greedy_search_sentence[0])
                reference_sent = ' '.join([str(k) for k in target_sentence])
                sampled_sent = ' '.join([str(k) for k in sampled_sentence[0]])
                sampled_sentence_r_values.append(reward_function(reference_sent, sampled_sent))
                greedy_sent = ' '.join([str(k) for k in greedy_search_sentence[0]])
                greedy_sentence_r_values.append(reward_function(reference_sent, greedy_sent))

            to_return = {
                'train_op': summarizationModel.train_op,
                'pgen_loss': summarizationModel._pgen_loss,
                'rl_loss': summarizationModel._rl_loss,
                'loss' : summarizationModel.loss
            }
            to_return['s_r'] = summarizationModel._sampled_sentence_r_values
            to_return['g_r'] = summarizationModel._greedy_sentence_r_values

            feed_dict[summarizationModel._sampled_sentence_r_values] = sampled_sentence_r_values
            feed_dict[summarizationModel._greedy_sentence_r_values] = greedy_sentence_r_values
            feed_dict[summarizationModel._eta] = 0.5
            res = sess.run(to_return, feed_dict)

            print('step : {0},pgen_loss : {1}, rl_loss : {2}, loss : {3}, reward : {4}'.format(step, res['pgen_loss'], res['rl_loss'], res['loss'],
                                                                                        np.sum(res['s_r']-res['g_r'])
                                                                                        ))
            if step % FLAGS.eval_step == 0:
                eval_ = run_rl_eval(summarizationModel, val_batcher, sess, 0.5)
                if eval_ < eval_loss:
                    if not os.path.exists(FLAGS.checkpoint):os.mkdir(FLAGS.checkpoint)
                    saver.save(sess, save_path=os.path.join(FLAGS.checkpoint,'model_{0}_{1}.ckpt'.format(step,eval_)))
                    eval_loss = eval_
                    patient = FLAGS.patient
                print('eval loss : ',eval_loss)
                if patient < 0:
                    break

                if eval_ - eval_loss > FLAGS.threshold:
                    patient -= 1

def decode(test_path,rl):
    sess = tf.Session(config=get_config())
    if FLAGS.beam == True:
        FLAGS.batch_size = FLAGS.beam_size
    FLAGS.max_dec_steps=1
    print('batch size ' , FLAGS.batch_size)
    #if rl == False:
    summarizationModel = PointerNet(FLAGS, vocab)
    #elif rl==True:
    #    if FLAGS.gamma > 0:
    #        import rl_model_gamma
    #        summarizationModel = rl_model_gamma.RLNet(FLAGS, vocab)
    #    else:
    #        import rl_model
    #        summarizationModel = rl_model.RLNet(FLAGS, vocab)
    summarizationModel.build_graph()
    saver = tf.train.Saver()
    best_model=load_best_model(FLAGS.restore_path)
    print('best model : {0}'.format(best_model))
    saver.restore(sess, save_path=best_model)
    counter = 0
    batcher = Batcher(test_path, vocab, FLAGS, single_pass=FLAGS.single_pass, decode_after=FLAGS.decode_after)
    batches = batcher.fill_batch_queue(is_training=False)  # 1 example repeated across batch
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    if FLAGS.beam == False:
        for batch in batches:
            article = batch.original_articles[0]
            original_abstract_sents = batch.original_abstracts_sents # list of strings
            #print('*****************start**************')
            best_hyps = beam_search.run_greedy_search(sess, summarizationModel, vocab, batch)
            output_ids = [[int(t) for t in best_hyp.tokens[1:]] for best_hyp in best_hyps]
            decoded_words = data.outputids2words_greedy(output_ids, vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
            decoded_words = remove_stop_index(decoded_words, data)
            write_for_rouge_greedy(original_abstract_sents, decoded_words, article, counter, FLAGS.dec_path, FLAGS.ref_path, FLAGS.all_path)  # write ref summary and decoded summary to file, to eval with pyrouge later
            counter += FLAGS.batch_size  # this is how many examples we've decoded
            print('counter ... ', counter)
            if counter % (5*64) == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    else:
        for batch in batches:
            article = batch.original_articles[0]
            original_abstract_sents = batch.original_abstracts_sents[0] # list of strings
            #print('*****************start**************')
            best_hyps = beam_search.run_beam_search(sess, summarizationModel, vocab, batch)
            #print('best hyp : {0}'.format(best_hyp))
            output_ids = [int(t) for t in best_hyps.tokens[1:]]
            decoded_words = data.outputids2words_beam(output_ids, vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            #decoded_words = ' '.join(decoded_words)
            write_for_rouge_beam(original_abstract_sents, decoded_words, article, counter, FLAGS.dec_path, FLAGS.ref_path, FLAGS.all_path)  # write ref summary and decoded summary to file, to eval with pyrouge later
            counter += 1  # this is how many examples we've decoded
            print('counter ... ', counter)
            if counter % 100 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))



if __name__ == '__main__':
    if FLAGS.attn_gamma != 0:
        from model_v2 import PointerNet
        print('import model_v2')
    if FLAGS.mode=='train' and FLAGS.rl==True:
        #run_rl_training()
        run_rl_training_gamma(FLAGS, vocab)
    elif FLAGS.mode=='train' and FLAGS.rl==False:
        run_training()
    elif FLAGS.mode=='decode':
        decode(FLAGS.data_path, FLAGS.rl)
    else:
        print('lalalala.......')
