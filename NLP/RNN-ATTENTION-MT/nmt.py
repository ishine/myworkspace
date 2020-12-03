import sys, os, time
import importlib
importlib.reload(sys)
from input_data import DataLoader, bacth_yield, padding
import argparse
import tensorflow as tf 
import utils
from tensorflow.contrib.rnn import BasicRNNCell, LSTMCell
from tensorflow.python.layers import core as layers_core
import sacrebleu
import numpy as np
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    # 训练数据路径
    parser.add_argument('--train_src_path', type = str, default = './data/train.zh.tok')
    parser.add_argument('--train_target_path', type = str, default = './data/train.en.tok')
    # 验证数据路径
    parser.add_argument('--valid_src_path', type = str, default = './data/valid.zh.tok')
    parser.add_argument('--valid_target_path', type = str, default = './data/valid.en.tok')
    # 测试数据路径
    parser.add_argument('--test_src_path', type = str, default = './data/test.zh.tok')
    parser.add_argument('--test_target_path', type = str, default = './data/test.en.tok')
    # 测试结果输出路径
    parser.add_argument('--result_path', type = str, default = './result/translate.en.tok')
    # 日志记录路径
    parser.add_argument('--log_path', type = str, default = 'log.txt')
    # 训练轮数
    parser.add_argument('--epochs', type = int, default = 10)
    # batch_size
    parser.add_argument('--batch_size', type = int, default = 64)
    # embedding_dim
    parser.add_argument('--embedding_dim', type = int, default = 300)
    # hidden_dim
    parser.add_argument('--hidden_dim', type = int, default = 300)
    # learning_rate
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    # mode, 设置是训练模型train, 还是根据模型预测infer
    parser.add_argument('--mode', type = str, default = 'train', help = 'train / infer')
    # beam_width
    parser.add_argument('--beam_width', type = int, default = 2)
    # model_path
    parser.add_argument('--model_path', type = str, default = '')
    return parser.parse_args()

class NMT():
    def __init__(self, args, train_loader, valid_loader, test_loader):
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.add_placeholder()
        self.add_embedding()
        self.build_encoder()
        self.build_decoder()
        self.loss_op()
        self._init_op()
        # 日志记录
        self.logger = utils.get_logger(self.args.log_path)

    # 设置placeholder
    def add_placeholder(self):
        # encoder输入
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name = 'encoder_inputs')
        # encoder输入长度
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name = 'encoder_inputs_length')
        # decoder输入
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name = 'decoder_inputs')
        # decoder输出
        self.decoder_outputs = tf.placeholder(tf.int32, [None, None], name = 'decoder_outputs')
        # decoder输入\输出长度
        self.decoder_length = tf.placeholder(tf.int32, [None], name = 'decoder_length')
        # 最大输出序列长度
        self.max_decoder_length = tf.reduce_max(self.decoder_length, name = 'max_decoder_length')
        # 设置dropout
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    # 设置共享embedding层
    def add_embedding(self):
        with tf.variable_scope('embedding'):
            # 输入层嵌入
            # shape(src_vocab_size, embedding_dim)
            self.embeddings_zh = tf.Variable(tf.random_uniform([self.train_loader.src_vocab_size, self.args.embedding_dim], -1.0, 1.0), name = 'embeddings_zh')
            
            # 输出层嵌入
            # shape(target_vocab_size, embedding_dim)
            self.embeddings_en = tf.Variable(tf.random_uniform([self.train_loader.target_vocab_size, self.args.embedding_dim], -1.0, 1.0), name = 'embeddings_en')

    # 构建编码器
    def build_encoder(self):
        with tf.variable_scope('encoder'):
            # 查找词向量
            # shape(None, None, embedding_dim)
            encoder_emb_in = tf.nn.embedding_lookup(self.embeddings_zh, self.encoder_inputs)
            # dropout
            encoder_emb_in = tf.nn.dropout(encoder_emb_in, self.dropout)
            # 本来使用的BasicRNNCell, 在训练过程中会梯度爆炸, 改成LSTMCell
            encoder_cell = LSTMCell(self.args.hidden_dim)
            # encoder_outputs存放着隐藏层的输出，encoder_state是最后一次隐藏层的输出
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                                         inputs = encoder_emb_in,
                                                                         sequence_length = self.encoder_inputs_length,
                                                                         dtype = tf.float32)
    # 构建解码器
    def build_decoder(self):
        with tf.variable_scope('decoder') as scope:
            # 定义BasicRNNCell decoder_cell
            decoder_cell, decoder_initial_state = self._build_decoder_cell()

            # 解码器输出的全连接层
            # shape(hidden_dim, target_vocab_size)
            projection_layer = layers_core.Dense(self.train_loader.target_vocab_size, use_bias = False)
            maximum_iterations = self.max_decoder_length

            if self.args.mode == 'train':
                # 查找词向量
                # shape(None, None, embedding_dim)
                decoder_emb_in = tf.nn.embedding_lookup(self.embeddings_en, self.decoder_inputs)
                # dropout
                decoder_emb_in = tf.nn.dropout(decoder_emb_in, self.dropout)
              
                # Helper
                # 接收参数
                # inputs: shape(batch_size, seqlen, embedding_dim)
                # sequence_length: 真实的句子长度
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = decoder_emb_in,
                                                                    sequence_length = self.decoder_length)
                # 解码器
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,
                                                                   helper = training_helper,
                                                                   initial_state = decoder_initial_state,
                                                                   output_layer = projection_layer)
                # 解码
                decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder = training_decoder, scope = scope)

                self.decoder_sample_id = decoder_outputs.sample_id
                self.logits = decoder_outputs.rnn_output
                self.decoder_final_context_state = final_context_state

            elif self.args.mode == 'infer':
                start_tokens = tf.ones([self.args.batch_size,], tf.int32) * self.train_loader.target_word2id['<START>']
                end_token = self.train_loader.target_word2id['<END>']
                # 解码器, 使用BeamSearch方法
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = decoder_cell,
                                                                            embedding = self.embeddings_en,
                                                                            start_tokens = start_tokens,
                                                                            end_token = end_token,
                                                                            initial_state = decoder_initial_state,
                                                                            beam_width = self.args.beam_width,
                                                                            output_layer = projection_layer)
                # 解码
                decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder = inference_decoder,
                                                                                            maximum_iterations = maximum_iterations,
                                                                                            scope = scope)
                
                self.logits = tf.no_op()
                self.predicted_sample_id = decoder_outputs.predicted_ids

    def _build_decoder_cell(self):
        beam_width = self.args.beam_width
        # memory保存着前面隐藏层的输出，用于做attention计算
        # shape(None, None, hidden_dim)
        # encoder_state是encoder最后隐藏层输出
        memory, encoder_state, encoder_inputs_length = self.encoder_outputs, self.encoder_state, self.encoder_inputs_length


        # beam search是神经序列生成模型最基本的解码算法
        if self.args.mode == 'infer' and beam_width > 0:
            # 将batch内的每个样本复制beam_width次
            memory = tf.contrib.seq2seq.tile_batch(memory, multiplier = beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier = beam_width)
            encoder_inputs_length = tf.contrib.seq2seq.tile_batch(encoder_inputs_length, multiplier = beam_width)
            batch_size = self.args.batch_size * beam_width
        else:
            batch_size = self.args.batch_size

        # attention模块
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.args.hidden_dim, memory = memory,
                                                                memory_sequence_length = encoder_inputs_length)
        # 本来使用的BasicRNNCell, 在训练过程中会梯度爆炸, 改成LSTMCell
        basic_cell = LSTMCell(self.args.hidden_dim)
        # 应该是basic_cell和attention模块结合, 实现attention机制
        cell = tf.contrib.seq2seq.AttentionWrapper(basic_cell, attention_mechanism,
                                                   attention_layer_size = self.args.hidden_dim,
                                                   output_attention = True,
                                                   name = 'attention')
        # 定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
        decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state = encoder_state)
        return cell, decoder_initial_state

    def loss_op(self):
        if self.args.mode == 'train':
            max_times = self.max_decoder_length
            # 计算交叉熵
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.decoder_outputs,
                                                                      logits = self.logits)
            # 设置mask, 屏蔽掉padding部分数据, 是二进制向量
            target_weights = tf.sequence_mask(self.decoder_length, max_times, dtype = self.logits.dtype)
            # 定义loss
            self.train_loss = (tf.reduce_sum(crossent * target_weights) / self.args.batch_size)
            # 设置优化方法
            optim = tf.train.AdamOptimizer(learning_rate = self.args.learning_rate)
            self.train_op = optim.minimize(self.train_loss)


    # 初始化变量
    def _init_op(self):
        self.init_op = tf.global_variables_initializer()

    # 模型训练
    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config = config) as sess:
            sess.run(self.init_op)
            for epoch in range(self.args.epochs):
                self.run_one_epoch(sess, epoch, saver)
    # 训练一轮
    def run_one_epoch(self, sess, epoch, saver):
        num_batches = (len(self.train_loader.data) + self.args.batch_size - 1) // self.args.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        train_batches = bacth_yield(self.train_loader.data, self.args.batch_size, shuffle = True)
        for step, (src, target_in, target_out) in enumerate(train_batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            # 填充
            encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_outputs, decoder_length = padding(src, target_in, target_out, pad_mark = 3)
            # 构造馈送的数据
            feed_dict = {
                self.encoder_inputs: encoder_inputs,
                self.encoder_inputs_length: encoder_inputs_length,
                self.decoder_inputs: decoder_inputs,
                self.decoder_outputs: decoder_outputs,
                self.decoder_length: decoder_length,
                self.dropout: 0.5
            }
            # 计算loss
            _, loss_train = sess.run([self.train_op, self.train_loss], feed_dict = feed_dict)

            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:

                self.logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, 
                                                                                   step + 1,loss_train, step_num))
            if step + 1 == num_batches:
                saver.save(sess, self.args.model_path, global_step = step_num)

    def test(self):
        self.logger.info('===========test===========')
        all_candidates = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            # 载入模型
            saver.restore(sess, self.args.model_path)

            test_batches =  bacth_yield(self.test_loader.data, self.args.batch_size, shuffle = False)
            for i, (src, target_in, target_out) in enumerate(test_batches):
                sys.stdout.write(' evaluating: #{} batch'.format(i + 1) + '\r')
                encoder_inputs, encoder_inputs_length, _, _, decoder_length = padding(src, target_in, target_out, pad_mark = 3)
                # 构造馈送的数据
                feed_dict = {
                    self.encoder_inputs: encoder_inputs,
                    self.encoder_inputs_length: encoder_inputs_length,
                    self.decoder_length: decoder_length,
                    self.dropout: 1.0
                }
                # 模型预测
                predict = sess.run([self.predicted_sample_id], feed_dict=feed_dict)
                predict = np.array(predict)
                temp = predict[0]
                for item in temp:
                    sent = []
                    # beam search选择概率最大的一句输出
                    for beam_item in item:
                        most_common_res = Counter(list(beam_item)).most_common(1)[0][0]
                        sent.append(most_common_res)
                    # 去除END_ID
                    if utils.END_ID in sent:
                        sent = sent[:sent.index(utils.END_ID)]
                    # 从单词id还原出单词
                    sent_word = utils.id2sentence(sent, self.train_loader.target_id2word)
                    all_candidates.append(sent_word)

            # 讲预测的结果写入infer.en.tok中
            with open('infer.en.tok', 'w') as f:
                candicates = [' '.join(line) + '\n' for line in all_candidates]
                f.writelines(candicates)
            # 将参考的结果写入ref.en.tok中
            with open('ref.en.tok', 'w') as f:
                target_lines = [' '.join(line) + '\n' for line in self.test_loader.target_lines]
                # 由于测试按batch_size划分, 导致原始的测试数据部分丢失, 于是使用切片，只保留与candicates同样长度的部分。
                f.writelines(target_lines[:len(candicates)])
            # 逐句测试
            bleuscore = []
            for sys, ref in zip(all_candidates, self.test_loader.target_lines):
                sys = ' '.join(sys)
                ref = ' '.join(ref)
                bleu = sacrebleu.corpus_bleu(sys, ref)
                bleuscore.append(bleu.score)
            # 求bleuscore均值
            print(np.mean(bleuscore))
            print(len(bleuscore))
            self.logger.info('bleuscore: %d' %np.mean(bleuscore))
        



        












if __name__ == '__main__':
    args = parse_args()

    # 训练数据
    train_loader = DataLoader(args.train_src_path, args.train_target_path,3)
    # 验证数据
    valid_loader = DataLoader(args.valid_src_path, args.valid_target_path,3)
    # 测试数据
    test_loader = DataLoader(args.test_src_path, args.test_target_path,3)


    if args.mode == 'train':
        print ("\n训练模式...")
        timestamp = str(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
        args.model_path = os.path.join('.', "model", timestamp + "/")
        if not os.path.exists(args.model_path): os.makedirs(args.model_path)
    else:
        timestamp = args.model_path
        args.model_path = os.path.join('.', "model", timestamp + "/")
        print ("\n加载的模型是：", timestamp, '\n')

    if args.mode == 'train':
        model = NMT(args, train_loader, valid_loader, test_loader)
        model.train()
    elif args.mode == 'infer':
        args.model_path = tf.train.latest_checkpoint(args.model_path)
        model = NMT(args, train_loader, valid_loader, test_loader)
        print ("=============================")
        model.test()



