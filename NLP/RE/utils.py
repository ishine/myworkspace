import os 
import json 
import numpy as np
import tensorflow as tf



def load_embedding(path, word_dim):
    """
    加载预训练词向量
    :param path: 预训练词向量的路径
    :param word_dim: 预训练词向量的维数
    :return: word2id和word_vec
    """
    word2id = dict()
    word_vec = list()

    word2id['PAD'] = len(word2id)
    word2id['UNK'] = len(word2id)

    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip().split()
            if len(line) != word_dim + 1:
                continue
            word2id[line[0]] = len(word2id)
            word_vec.append(np.asarray(line[1:], dtype=np.float32))
    special_emb = np.random.uniform(-1, 1, (2, word_dim))
    special_emb[0] = 0
    word_vec = np.concatenate((special_emb, word_vec), axis=0)
    word_vec = word_vec.astype(np.float32).reshape(-1, word_dim)
    return word2id, word_vec

def load_relation(path):
    """
    加载关系-id映射
    :param path: 关系-id映射文件所在目录
    :return: rel2id, id2rel, len(rel2id)
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
    def __init__(self, rel2id, word2id, config):
        self.rel2id = rel2id
        self.word2id = word2id
        self.data_dir = config.data_dir
        self.max_len = config.max_len
        self.pos_dis = config.pos_dis
        self.batch_size = config.batch_size

    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    def __get_relative_pos(self, x, entity_pos):
        if x < entity_pos[0]:
            return self.__get_pos_index(x-entity_pos[0])
        elif x > entity_pos[1]:
            return self.__get_pos_index(x-entity_pos[1])
        else:
            return self.__get_pos_index(0)

    def __symbolize_sentence(self, e1_pos, e2_pos, sentence):
        mask = [1] * len(sentence)
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1]+1):
                mask[i] = 2
            for i in range(e2_pos[1]+1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1]+1):
                mask[i] = 2
            for i in range(e1_pos[1]+1, len(sentence)):
                mask[i] = 3

        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['UNK']))
            pos1.append(self.__get_relative_pos(i, e1_pos))
            pos2.append(self.__get_relative_pos(i, e2_pos))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])

                pos1.append(self.__get_relative_pos(i, e1_pos))
                pos2.append(self.__get_relative_pos(i, e2_pos))
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(4, self.max_len))
        return unit
    
    def __get_data(self, filename, shuffle=False):
        data = []
        labels =[]
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['sentence']
                e1_pos = (line['subj_start'], line['subj_end'])
                e2_pos = (line['obj_start'], line['obj_end'])
                label_idx = self.rel2id[label]

                one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence)
                data.append(one_sentence)
                labels.append(label_idx)
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        if shuffle:
            dataset = dataset.shuffle(len(data))
        dataset = dataset.batch(self.batch_size)
        return dataset



    def get_train(self):
        train_file = os.path.join(self.data_dir, 'train.json')
        return self.__get_data(train_file, shuffle=True)
    def get_dev(self):
        dev_file = os.path.join(self.data_dir, 'test.json')
        return self.__get_data(dev_file, shuffle=False)
    def get_test(self):
        test_file = os.path.join(self.data_dir, 'test.json')
        return self.__get_data(test_file, shuffle=False)



if __name__ == '__main__':
    #load_embedding('./embedding/glove.6B.300d.txt', 300)
    from config import Config
    config = Config()

    word2id, word_vec = load_embedding(config.embedding_path, config.word_dim)
    #print(word2id)
    rel2id, id2rel, class_num = load_relation(config.data_dir)
    #print(rel2id)
    loader = DataLoader(rel2id, word2id, config)
    train_loader = loader.get_train().repeat(config.epoch)
    dev_loader = loader.get_dev()
    test_loader = loader.get_test()
    for data, label in test_loader:
        sent = data[:,0,:]
        pos1 = data[:,1,:]
        pos2 = data[:,2,:]
        mask = data[:,3,:]
        
        print(data[:,0,:])
        print(sent.shape)
        sent = tf.reshape(sent, [-1,100])
        print(sent.shape)
        print(data.shape)
        break

