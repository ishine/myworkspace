import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential



class CRCNN():
    def __init__(self, word_vec, class_num, config):
        self.word_vec = word_vec
        self.class_num = class_num 

        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.pos_dim = config.pos_dim 
        self.pos_dis = config.pos_dis

        self.dropout_value = config.dropout 
        self.filter_num = config.filter_num
        self.window = config.window

        self.dim = self.word_dim + 2 * self.pos_dim

    def add_placeholder(self):
        """
        添加输入层
        """
        self.sent = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name='sent')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
 
    def add_embedding(self):
        """
        添加隐藏层
        """
        with tf.variable_scope('embedding'):
            # 词嵌入层
            self.word_embedding = tf.Variable(tf.convert_to_tensor(self.word_vec), name='word_embedding')
            # pos嵌入层
            self.pos1_embedding = tf.Variable(tf.random_uniform([2 * self.pos_dis + 3, self.pos_dim], -1.0, 1.0), name='pos1_embedding')
            self.pos2_embedding = tf.Variable(tf.random_uniform([2 * self.pos_dis + 3, self.pos_dim], -1.0, 1.0), name='pos2_embedding')
    def builder_cnn(self):
        sent_input = tf.nn.embedding_lookup(self.word_embedding, self.sent)
        pos1_input = tf.nn.embedding_lookup(self.pos1_embedding, self.pos1)
        pos2_input = tf.nn.embedding_lookup(self.pos2_embedding, self.pos2)

        conv_input = tf.concat([sent_input, pos1_input, pos2_input], axis=-1)
        conv_input = tf.expand_dims(conv_input, -1, name='input')
        with tf.variable_scope('conv') as scope:
            pool_tensors = []
            
if __name__ == '__main__':
    pass
