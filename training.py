from rl_model_gamma import RLNet
from batcher import Batcher
import tensorflow as tf
from helper import make_feed_dict,get_config
from helper import reader_params,load_best_model
import os
from helper import reward_function,loading_variable
import numpy as np

from eval import run_eval

def run_rl_eval_gramma(model, batcher, sess, eta):
    loss = 0
    batches = batcher.fill_batch_queue(is_training=False)
    for batch in batches:
        feed_dict = make_feed_dict(model, batch)
        feed_dict[model._eta] = eta
        eloss = sess.run(model._pgen_loss, feed_dict)
        loss += eloss
    return loss



def compute_reward(sampled, target):
    r = []
    for i in range(len(sampled)):
        r.append(reward_function(' '.join([str(k) for k in target]), ' '.join([str(k) for k in sampled[:i+1]])))
    return r


def run_rl_training_gamma(FLAGS, vocab):
    summarizationModel = RLNet(FLAGS, vocab)
    summarizationModel.build_graph()
    batcher = Batcher(FLAGS.data_path, vocab, FLAGS, single_pass=FLAGS.single_pass, decode_after=FLAGS.decode_after)
    val_batcher = Batcher(FLAGS.val_data_path, vocab, FLAGS, single_pass=FLAGS.single_pass,
                          decode_after=FLAGS.decode_after)
    sess = tf.Session(config=get_config())
    saver = tf.train.Saver(max_to_keep=100)
    if FLAGS.restore_rl_path:
        print('restore rl model...')
        saver.restore(sess, FLAGS.restore_rl_path)
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(
            loading_variable([v for v in tf.trainable_variables()], reader_params(load_best_model(FLAGS.restore_path))))
        print('loading params...')
    epoch = FLAGS.epoch
    step = 0
    patient = FLAGS.patient
    eval_max_reward = -float('inf')
    while epoch > 0:
        batches = batcher.fill_batch_queue()
        for batch in batches:
            step += 1
            sampled_sentence_r_values = []
            greedy_sentence_r_values = []
            feed_dict = make_feed_dict(summarizationModel, batch)

            to_return = {
                'sampled_sentences': summarizationModel.sampled_sentences,

            }
            ret_dict = sess.run(to_return, feed_dict)
            Rs = []
            # calculate reward
            for sampled_sentence, target_sentence in zip(ret_dict['sampled_sentences'],batch.target_batch):
              #  print('sampled : ',sampled_sentence)
              #  print('target : ', target_sentence)
                reward = compute_reward(sampled_sentence[0], target_sentence)
                R = 0
                R_l = []
                for r in reward[::-1]:
                    R = r + FLAGS.gamma * R
                    R_l.insert(0,R)
                #avg = np.mean(R_l)
                #R_l = list(map(lambda a:a-avg, R_l))
                Rs.append(R_l)
            to_return = {
                'train_op': summarizationModel.train_op,
                'pgen_loss': summarizationModel._pgen_loss,
                'rl_loss': summarizationModel._rl_loss,
                'loss': summarizationModel.loss
            }
            to_return['reward'] = summarizationModel._reward

            
            feed_dict[summarizationModel._reward] = Rs
            feed_dict[summarizationModel._eta] = 0.5
            res = sess.run(to_return, feed_dict)

            print('step : {0}, pgen_loss : {1}, rl_loss : {2}, loss : {3}, reward : {4}'.format(step,res['pgen_loss'], res['rl_loss'],
                                                                                    res['loss'],
                                                                                    np.mean(res['reward'],axis=0)[0]
                                                                                    ))
            if step % FLAGS.eval_step == 0:
                #eval_ = run_rl_eval_gramma(summarizationModel, val_batcher, sess, 0.5)
                eval_reward = run_eval(summarizationModel, val_batcher, sess)
                if eval_reward > eval_max_reward:
                    if not os.path.exists(FLAGS.checkpoint): os.mkdir(FLAGS.checkpoint)
                    saver.save(sess, save_path=os.path.join(FLAGS.checkpoint, 'model_{0}_{1}.ckpt'.format(step, eval_reward)))
                    eval_reward = eval_max_reward
                    patient = FLAGS.patient
                print('eval loss ', eval_max_reward)
                if patient < 0:
                    break

                if eval_max_reward - eval_reward > FLAGS.threshold:
                    patient -= 1

