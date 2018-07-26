import tensorflow as tf
import time
from model import PointerNet
from helper import load_best_model
from batcher import Batcher
import numpy as np
from helper import get_config,write_for_rouge_beam
import beam_search
import data
import threading

def split_batches(batches, num):
    data_array =  np.array_split(np.array(batches),num)
    counter_array = [0]
    for i in range(1,len(data_array)):
        counter_array.append(counter_array[i-1] + len(data_array[i]))
    return data_array, counter_array

def thread_decode(test_path,vocab, FLAGS):
    sess = tf.Session(config=get_config())
    if FLAGS.beam == True:
        FLAGS.batch_size = FLAGS.beam_size
    FLAGS.max_dec_steps=1
    print('batch size ' , FLAGS.batch_size)

    summarizationModel = PointerNet(FLAGS, vocab)
    summarizationModel.build_graph()
    saver = tf.train.Saver()
    COORD = tf.train.Coordinator()
    best_model=load_best_model(FLAGS.restore_path)
    print('best model : {0}'.format(best_model))
    saver.restore(sess, save_path=best_model)
    batcher = Batcher(test_path, vocab, FLAGS, single_pass=FLAGS.single_pass, decode_after=FLAGS.decode_after)
    batches = batcher.cpu_fill_batch_queue()  # 1 example repeated across batch
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    #切分数据
    split_datas, count_array = split_batches(batches, FLAGS.work_num)
    print("len batches : {0}".format(len(batches)))
    assert len(split_datas) == FLAGS.work_num
    work_threads = []
    for i in range(FLAGS.work_num):
        job = lambda : do(split_datas[i], summarizationModel, vocab, sess, FLAGS, count_array[i],i)
        t = threading.Thread(target=job)
        t.start()
        work_threads.append(t)
        print('work : {0}'.format(i))
    COORD.join(work_threads)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def do(batches, model, vocab, sess, FLAGS, counter,i):
    print('work : {0}, len : {1}'.format(i, len(batches)))
    for batch in batches:
        article = batch.original_articles[0]
        original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings
        best_hyps = beam_search.run_beam_search(sess, model, vocab, batch)
        output_ids = [int(t) for t in best_hyps.tokens[1:]]
        decoded_words = data.outputids2words_beam(output_ids, vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        write_for_rouge_beam(original_abstract_sents, decoded_words, article, counter, FLAGS.dec_path, FLAGS.ref_path,
                         FLAGS.all_path)  # write ref summary and decoded summary to file, to eval with pyrouge later
        counter += 1  # this is how many examples we've decoded
        #print('counter ... ', counter)
        #if counter % 100 == 0:
        #    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
