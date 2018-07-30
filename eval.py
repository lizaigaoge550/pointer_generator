from helper import make_feed_dict,reward_function
import numpy as np

def get_reward(sess, model, feed_dict, batch):
    to_return = {
       'greedy_search_sentences' : model.greedy_search_sentences
    }
    ret_dict = sess.run(to_return, feed_dict)
    reward = []
    for greedy_search_sentence, target_sentence in zip(ret_dict['greedy_search_sentences'], batch.target_batch):
        assert len(target_sentence) == len(greedy_search_sentence[0])
        reference_sent = ' '.join([str(k) for k in target_sentence])
        greedy_sent = ' '.join([str(k) for k in greedy_search_sentence[0]])
        reward.append(reward_function(reference_sent, greedy_sent))
    return reward

def run_eval(model, batcher, sess):
    print('eval start ......')
    reward = []
    batches = batcher.fill_batch_queue(is_training=False)
    for batch in batches:
        feed_dict = make_feed_dict(model, batch)
        #eloss = sess.run(model.loss, feed_dict)
        reward.extend(get_reward(sess, model, feed_dict, batch))
        #loss += eloss
    return np.mean(reward)