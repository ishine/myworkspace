import sys, os, time
import importlib
import tensorflow as tf 
from tensorflow.contrib.rnn import LSTMCell

from utils import *
from config import Config


class AttLSTM():
    def __init__(self, word_vec, class_num, config, model_path, id2rel):
        # 预训练词向量
        self.word_vec = word_vec
        # 关系类别数目
        self.class_num = class_num 
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # l2正则化系数
        self.l2_reg_lambda = config.l2
        # epoch
        self.epoch = config.epoch
        # batch_size
        self.batch_size = config.batch_size
        # learning rate
        self.lr = config.lr
        self.model_path = model_path
        # id到关系的映射
        self.id2rel = id2rel

        self.add_placeholder()
        self.add_embedding()
        self.build()
        self._init_op()
    def add_placeholder(self):
        """
        添加输入层
        """
        self.sent = tf.placeholder(dtype=tf.int64, shape=[None, None], name='sent')
        self.length = tf.placeholder(tf.int64, shape=[None], name='length')
        self.label = tf.placeholder(dtype=tf.int64, shape=[None], name='label')
        
    def add_embedding(self):
        """
        添加隐藏层
        """
        with tf.variable_scope('embedding'):
            # 词嵌入层
            embeddings = tf.Variable(tf.convert_to_tensor(self.word_vec, tf.float32), name='embeddings')
        # 词嵌入向量
        self.word_embedding = tf.nn.embedding_lookup(embeddings, self.sent, name='word_embedding')

    def build(self):
        """
        构建网络
        """
        # BiLSTM
        with tf.variable_scope("bilstm"):
            # 前向LSTM层
            cell_fw = LSTMCell(self.hidden_size)
            # 后向LSTM层
            cell_bw = LSTMCell(self.hidden_size)
            # 双向LSTM，BiLSTM
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw,
                                                                  cell_bw = cell_bw,
                                                                  inputs = self.word_embedding,
                                                                  sequence_length = self.length,
                                                                  dtype = tf.float32)
            self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])
        # Attention
        with tf.variable_scope("attention"):
            self.attn, self.alphas = self.attention(self.rnn_outputs)
        # 全连接
        with tf.variable_scope("output"):
            self.logits = tf.layers.dense(self.attn, self.class_num)
            self.predictions = tf.argmax(self.logits, 1, name='predictions')
        # loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2
        # 计算准确度
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        # 设置优化方法
        optim = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train_op = optim.minimize(self.loss)

    # 初始化变量
    def _init_op(self):
        self.init_op = tf.global_variables_initializer()
    
    # 模型训练
    def train(self, train_data):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config = config) as sess:
            sess.run(self.init_op)
            for epoch in range(self.epoch):
                self.run_one_epoch(sess, epoch, train_data, saver)
    # 训练一轮
    def run_one_epoch(self, sess, epoch, train_data, saver):
        num_batches = (len(train_data) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        train_batches = bacth_yield(train_data, self.batch_size, True)
        for step, (sent, label) in enumerate(train_batches):
            seq_list, seq_len_list = pad_sequences(sent)
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict = {
                self.sent:seq_list,
                self.length:seq_len_list,
                self.label:label
            }
            # 计算loss
            _, loss_train, accuracy = sess.run([self.train_op, self.loss, self.accuracy], feed_dict = feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                print('{} epoch {}, step {}, loss: {:.4}, accuracy: {:.4}, global_step: {}'.format(start_time, epoch + 1, 
                                                                                   step + 1,loss_train, accuracy, step_num))
            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step = step_num)
    def test(self, test_data):
        all_predict = []
        reference = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            # 载入模型
            saver.restore(sess, self.model_path)
            test_batches = bacth_yield(test_data, self.batch_size, False)
            for i, (sent, label) in enumerate(test_batches):
                sys.stdout.write('evaluating: #{} batch'.format(i + 1) + '\r')
                seq_list, seq_len_list = pad_sequences(sent)
                feed_dict = {
                    self.sent:seq_list,
                    self.length:seq_len_list,
                    self.label:label
                }
                predict = sess.run([self.predictions], feed_dict=feed_dict)
                temp = predict[0]
                for item in temp:
                    all_predict.append(id2rel[item])
                for l in label:
                    reference.append(id2rel[l])
        # 将预测结果写入predict.txt中
        with open('predict.txt', 'w') as f:
            all = [ "%d\t%s" %(i, str) for i, str in enumerate(all_predict)]
            f.writelines('\n'.join(all))
            f.write('\n')
        # 将参考结果写入reference.txt中
        with open('reference.txt', 'w') as f:
            all = [ "%d\t%s" %(i, str) for i, str in enumerate(reference)]
            f.writelines('\n'.join(all))
            f.write('\n')
    
    def attention(self, inputs):
        # Trainable parameters
        hidden_size = inputs.shape[2].value
        u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

        with tf.name_scope('v'):
            v = tf.tanh(inputs)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        # Final output with tanh
        output = tf.tanh(output)

        return output, alphas



if __name__ == '__main__':
    config = Config()

    rel2id, id2rel, class_num = load_relation(config.data_dir)
    loader = DataLoader(rel2id, config)
    word2id = loader.word2id

    word_vec = load_embedding(config.embedding_path, word2id, config.word_dim)

    train_data = loader.get_train()
    test_data = loader.get_test()
    # mode = 1时是训练模式
    # mode = 1
    # mode = 0时是测试模式
    mode = 0
    if mode:
        '''训练模式'''
        print ("\n训练模式...")
        timestamp = str(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
        model_path = os.path.join('.', "model", timestamp + "/")
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
    else:
        '''测试模式'''
        timestamp = '2020-12-19 00-31-23'#手动修改被保存的模型路径
        model_path = os.path.join('.', "model", timestamp + "/")
        print ("\n加载的模型是：", timestamp, '\n')
    
    if mode:
        # 训练模型
        model = AttLSTM(word_vec, class_num, config, model_path, id2rel)
        model.train(train_data)
    else:
        # 测试模型
        model_path = tf.train.latest_checkpoint(model_path)
        model = AttLSTM(word_vec, class_num, config, model_path, id2rel)
        model.test(test_data)


