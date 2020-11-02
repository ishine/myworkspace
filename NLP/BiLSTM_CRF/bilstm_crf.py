#encoding:utf-8
import sys, os, time
import importlib
importlib.reload(sys)

from input_data import *

import numpy as np
import argparse
import logging
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

trainCorpusPath = './data/train_corpus.txt'
trainLabelPath = './data/train_label.txt'
testCorpusPath = './data/test_corpus.txt'
testLabelPath = './data/test_label.txt'
vocabPath = './data/char2id.pkl'
logPath = './result/log.txt'
result_path = './result'

def main():
    parser = argparse.ArgumentParser(description='BiLSTM-CRF')
    parser.add_argument('--batch_size', type=int, default=128, help='#sample of each minibatch')
    parser.add_argument('--epoch', type=int, default=10, help='#epoch of training')
    parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
    parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
    parser.add_argument('--num_tags', type=int, default = 7, help = 'tag number')
    args = parser.parse_args()

    # 训练数据
    trainData = read_data(trainCorpusPath, trainLabelPath)
    # 测试数据
    testData = read_data(testCorpusPath, testLabelPath)
    # 准备char2id字典
    char2id = read_dictionary(vocabPath)
    print(trainData[0])
    print(len(char2id))
    # 准备logger
    logger = get_logger(logPath)

    # 模型定义
    graph = tf.Graph()
    with graph.as_default():
        # 模型输入
        # X shape(?,?)
        input_data = tf.placeholder(tf.int32, shape = [None, None], name = 'input_data')
        # Y shape(?,?)
        targets = tf.placeholder(tf.int32, shape = [None, None], name = 'targets')
        # 相当于mask的作用, shape(?,)
        sequence_lengths = tf.placeholder(tf.int32, shape = [None], name = 'sequence_lengths')
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        # 模型参数
        with tf.variable_scope('embeddings'):
            # 字嵌入
            embeddings = tf.Variable(tf.random_uniform([len(char2id), args.embedding_dim], -1.0, 1.0), name = 'embeddings')
            # 字嵌入向量 shape(?,?,embedding_dim)
            char_embeddings = tf.nn.embedding_lookup(embeddings, input_data, name = 'char_embeddings')
            # dropout
            char_embeddings = tf.nn.dropout(char_embeddings, dropout_pl)
        # BiLSTM层
        with tf.variable_scope("bilstm"):
            # 前向LSTM层
            cell_fw = LSTMCell(args.hidden_dim)
            # 后向LSTM层
            cell_bw = LSTMCell(args.hidden_dim)
            # 双向LSTM，BiLSTM
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw,
                                                                                cell_bw = cell_bw,
                                                                                inputs = char_embeddings,
                                                                                sequence_length = sequence_lengths,
                                                                                dtype = tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis = -1)
            output = tf.nn.dropout(output, dropout_pl)
        # 预测输出
        with tf.variable_scope("proj"):
            W = tf.get_variable(name = "W",
                                shape = [2 * args.hidden_dim, args.num_tags],
                                initializer = tf.contrib.layers.xavier_initializer(),
                                dtype = tf.float32)
            b = tf.get_variable(name = "b",
                                shape = [args.num_tags],
                                initializer = tf.zeros_initializer(),
                                dtype = tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*args.hidden_dim])
            pred = tf.matmul(output, W) + b

            logits = tf.reshape(pred, [-1, s[1], args.num_tags])
        # loss
        log_likelihood, transition_params = crf_log_likelihood(inputs = logits,
                                                               tag_indices = targets,
                                                               sequence_lengths = sequence_lengths)
        loss = -tf.reduce_mean(log_likelihood)

        # 设置优化方法
        with tf.variable_scope("train_step"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate = args.lr)
            grads_and_vars = optim.compute_gradients(loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -args.clip, args.clip), v] for g, v in grads_and_vars]
            train_op = optim.apply_gradients(grads_and_vars_clip, global_step = global_step)


        saver = tf.train.Saver(tf.global_variables())  
    # 模型训练
    with tf.Session(graph = graph) as sess:
        tf.global_variables_initializer().run()

        for epoch in range(args.epoch):
            num_batches = len(trainData) // args.batch_size
            batches = batch_yield(trainData, args.batch_size, char2id, tag2labelDict, shuffle = True)

            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            #print(len(list(batches)))
            for step, (seqs, labels) in enumerate(batches):
                sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
                step_num = epoch * num_batches + step + 1
                _input_data, _sequence_lengths = pad_sequences(seqs)
                _targets, _ = pad_sequences(labels)
                feed_dict = {
                    input_data: _input_data,
                    sequence_lengths: _sequence_lengths,
                    targets: _targets,
                    dropout_pl: args.dropout
                }
                _, loss_train, step_num_ = sess.run([train_op, loss, global_step],
                                                        feed_dict = feed_dict)
                if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                    logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                            loss_train, step_num))
                break
            break
        logger.info('===========test===========')
        
        
        # 模型评估
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(testData, args.batch_size, char2id, tag2labelDict, shuffle = False):
            _word_ids, seq_len_list_ = pad_sequences(seqs)
            _labels_, _ = pad_sequences(labels)
            feed_dict = {
                    input_data: _word_ids,
                    sequence_lengths: seq_len_list_,
                    targets: _labels_,
                    dropout_pl: 1.0
                }
            _logits, _transition_params = sess.run([logits, transition_params],
                                                feed_dict = feed_dict)
            label_list_ = []
            for logit, seq_len in zip(_logits, seq_len_list_):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], _transition_params)
                label_list_.append(viterbi_seq)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        evaluate(logger, label_list, seq_len_list, testData, epoch)
        



def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def conlleval(label_predict, label_path, metric_path):
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics

def evaluate(logger, label_list, seq_len_list, data, epoch=None):

    label2tag = {}
    for tag, label in tag2labelDict.items():
        label2tag[label] = tag if label != 0 else label

    model_predict = []
    for label_, (sent, tag) in zip(label_list, data):
        tag_ = [label2tag[label__] for label__ in label_]
        sent_res = []
        if  len(label_) != len(sent):
            print(sent)
            print(len(label_))
            print(tag)
        for i in range(len(sent)):
            sent_res.append([sent[i], tag[i], tag_[i]])
        model_predict.append(sent_res)
    epoch_num = str(epoch+1) if epoch != None else 'test'
    label_path = os.path.join(result_path, 'label_' + epoch_num)
    metric_path = os.path.join(result_path, 'result_metric_' + epoch_num)
    for _ in conlleval(model_predict, label_path, metric_path):
        logger.info(_)

if __name__ == '__main__':
    main()
