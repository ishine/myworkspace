#encoding:utf-8
import sys
import importlib
importlib.reload(sys)
import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools
import random

tag2labelDict = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-LOC": 3, "I-LOC": 4,
    "B-ORG": 5, "I-ORG": 6
}

label2tagDict = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-LOC", 4: "I-LOC",
    5: "B-ORG", 6: "I-ORG"
}

def read_data(corpusPath, labelPath):
    data = []
    char_dict = collections.Counter()
    with open(corpusPath) as fr:
        corpusData = fr.readlines()
    with open(labelPath) as fr:
        labelData = fr.readlines()
    for corpus, label in zip(corpusData, labelData):
        sent = corpus.strip().split()
        tag = label.strip().split()
        assert len(sent) == len(tag), "len(sent) != len(tag)"
        data.append((sent, tag))
    return data

def vocab_build(vocabPath, data, min_frq):
    charCounts = collections.Counter()
    for sent, label in data:
        charCounts.update(sent)
    # charCounts统计字频，形式为{char1:frq1, char2:frq2, ....}
    vocabulary_inv = ['<PAD>'] + \
                     [x[0] for x in charCounts.most_common() if x[1] >= min_frq] +\
                     ['<UNK>']
    #vocabulary_inv取字频 >= min_frq的词，形式为[char1, char2, char3: ...]
    print(len(charCounts))
    print(len(vocabulary_inv))
    
    # {char:id, }
    char2id = {x: i for i, x in enumerate(vocabulary_inv)}
    # {id:char, }
    id2char = {v: k for k, v in char2id.items()}
    #print(char2id)
    #print(id2char)
    print(len(id2char))
    print(id2char[4768])
    with open(vocabPath, 'wb') as fw:
        cPickle.dump(char2id, fw)
    return char2id, id2char

def sentence2id(sent, char2id):
    return [char2id.get(w, 1) for w in sent]

def id2sentence(sentid, id2char):
    return [id2char.get(w, 1) for w in sentid]

def tag2label(tag, tag2lableDict):
    return [tag2labelDict.get(w,1) for w in tag]
'''
def label2tag(label, label2tagDict):
    return [label2tagDict.get(w,1) for w in label]
'''

def read_dictionary(vocabPath):
    with open(vocabPath, 'rb') as fr:
        char2id = cPickle.load(fr)
    print('vocab_size: ', len(char2id))
    return char2id

def random_embedding(char2id, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(char2id), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    print(embedding_mat.shape)
    return embedding_mat

def pad_sequences(sequences, pad_mark = 0):
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def batch_yield(data, batch_size, char2id, tag2labelDict, shuffle = False):
    if shuffle:
        random.shuffle(data)
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, char2id)
        label_ = tag2label(tag_, tag2labelDict)
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)
    # 不足一个batch_size的舍弃
    #if len(seqs) != 0:
        #yield seqs, labels

    

if __name__ == '__main__':
    corpusPath = './data/train_corpus.txt'
    labelPath = './data/train_label.txt'
    vocabPath = './data/char2id.pkl'
    data = read_data(corpusPath, labelPath)
    char2id, id2char = vocab_build(vocabPath, data, 0)
    '''
    sent0 = data[0][0]
    print('#SENT0', sent0)
    sent0id = sentence2id(sent0, char2id)
    print('#SETN02ID', sent0id)
    id2sent0 = id2sentence(sent0id, id2char)
    print('#ID2SENT0', id2sent0)
    read_dictionary(vocabPath)
    random_embedding(char2id, 64)
    batches = batch_yield(data, 2, char2id, tag2label, shuffle = False)
    print(max(map(lambda x : len(x), list(batches)[0][0])))
    '''
