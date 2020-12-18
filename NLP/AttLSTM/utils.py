import os 
import json 
import numpy as np
import tensorflow as tf
import sys
import collections
import random





def load_embedding(path, word2id, word_dim):
    """
    加载预训练词向量
    :param path: 预训练词向量的路径
    :param word2id
    :param word_dim: 预训练词向量的维数
    :return: word_vec
    """
    #initW = np.random.randn(len(word2id), word_dim).astype(np.float32) / np.sqrt(len(word2id))
    initW = np.random.uniform(-1, 1, (len(word2id), word_dim))
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            # 遍历预训练词向量
            line = line.strip().split()
            if len(line) != word_dim + 1:
                continue
            embedding = np.asarray(line[1:], dtype=np.float32)
            if line[0] in word2id.keys():
                # 如果word2id中存在单词line[0]
                # 则将line[0]中的预训练词向量替换initW中相应位置的词向量
                idx = word2id[line[0]]
                initW[idx] = embedding
    return initW

def load_relation(path):
    """
    加载关系-id映射
    :param path: 关系-id映射文件所在目录
    :return: rel2id, id2rel, class_num
    """
    relation_file = os.path.join(path, 'relation2id.txt')

    rel2id = {}
    id2rel = {}
    with open(relation_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            relation, id_s = line.strip().split()
            id_d = int(id_s)
            rel2id[relation] = id_d
            id2rel[id_d] = relation
    print(rel2id)
    return rel2id, id2rel, len(rel2id)

class DataLoader():
    def __init__(self, rel2id, config):
        self.rel2id = rel2id
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        train_file = os.path.join(self.data_dir, 'train.json')
        self.vocabulary, self.word2id = self.__vocab_build(train_file)


    def __vocab_build(self, filename, min_frq = 0):
        """
        从训练集数据中构建字典word2id
        """
        word_counts = collections.Counter()
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                sentence = line['sentence']
                word_counts.update(sentence)
        vocabulary = ['PAD', 'UNK'] + [x[0] for x in word_counts.most_common() if x[1] >= min_frq]
        # word:id字典
        word2id = {x: i for i, x in enumerate(vocabulary)}
        return vocabulary, word2id
    
    def __symbolize_sentence(self, sentence):
        """
        符号化句子，即word->id映射
        """
        words = []
        length = len(sentence)
        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['UNK']))
        return words
    
    def __get_data(self, filename):
        data = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['sentence']
                label_idx = self.rel2id[label]
                one_sentence = self.__symbolize_sentence(sentence)
                data.append((one_sentence, label_idx))
        return data
        
    def get_train(self):
        train_file = os.path.join(self.data_dir, 'train.json')
        return self.__get_data(train_file)
    def get_test(self):
        test_file = os.path.join(self.data_dir, 'test.json')
        return self.__get_data(test_file)

def bacth_yield(data, batch_size, shuffle = False):
    """
    数据分片
    """
    if shuffle:
        random.shuffle(data)
    sentence, labels = [], []

    for sent, label in data:
        if len(sentence) == batch_size:
            yield sentence, labels
            sentence, labels = [], []
        sentence.append(sent)
        labels.append(label)
    if len(sentence):
        yield sentence, labels

def pad_sequences(sequences, pad_mark = 0):
    """
    数据填充，返回填充后的句子，即原始的长度
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


if __name__ == '__main__':
    #load_embedding('./embedding/glove.6B.300d.txt', 300)
    from config import Config
    config = Config()

    #word2id, word_vec = load_embedding(config.embedding_path, config.word_dim)
    #print(word2id)
    rel2id, id2rel, class_num = load_relation(config.data_dir)
    #print(rel2id)
    loader = DataLoader(rel2id, config)
    train_data = loader.get_train()
    vocabulary = loader.vocabulary
    word2id = loader.word2id
    print(len(vocabulary), len(word2id))
    word_vec = load_embedding(config.embedding_path, word2id, config.word_dim)
    
    train_batches = bacth_yield(train_data, 2)
    for sent, label in train_batches:
        print(sent)
        print(label)
        seq_list, seq_len_list = pad_sequences(sent)
        print(seq_list)
        print(seq_len_list)
        break

