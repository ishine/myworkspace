import sys
import utils
import collections
import random



class DataLoader():
    def __init__(self, src_path, target_path, min_frq = 3):
        # 源文件路径
        self.src_path = src_path
        # 目标文件路径
        self.target_path = target_path
        # 词频阈值
        self.min_frq = min_frq

        # 源文件词表, word2id字典, id2word字典
        self.src_vocab, self.src_word2id, self.src_id2word, self.__src_lines= self.__vocab_build(self.src_path, self.min_frq)
        # 源数据词表大小
        self.src_vocab_size = len(self.src_vocab)
        # 目标文件词表, word2id字典, id2word字典
        self.target_vocab, self.target_word2id, self.target_id2word, self.__target_lines= self.__vocab_build(self.target_path, self.min_frq)
        self.target_lines = self.__target_lines
        # 目标数据词表大小
        self.target_vocab_size = len(self.target_vocab)
        # 获取句子的id表示, 其中__target_in在句首添加了START_ID, __target_out在句尾添加了END_ID
        self.__src, self.__target_in, self.__target_out = self.__get_id()

        self.data = list(zip(self.__src, self.__target_in, self.__target_out))

    # 建立词典
    def __vocab_build(self, filename, min_frq = 3):

        word_counts = collections.Counter()

        with open(filename, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        # 统计词频
        for line in lines:
            word_counts.update(line)
        # 词表，保留词频 >= min_frq的单词
        
        vocabulary_inv = ['<UNK>', '<START>', '<END>', '<PADDING>'] + \
                            [x[0] for x in word_counts.most_common() if x[1] >= min_frq]
        # word:id 字典
        word2id = {x: i for i, x in enumerate(vocabulary_inv)}
        # id:word 字典
        id2word = {i: x for i, x in enumerate(vocabulary_inv)}
        # 将词表，字典保存
        utils.pickle_dump(filename + '.pkl', (vocabulary_inv, word2id, id2word))
        return vocabulary_inv, word2id, id2word, lines
    # 句子中的单词替换成id
    def __get_id(self):
        src = [utils.sentence2id(sentence, self.src_word2id) + [utils.END_ID] for sentence in self.__src_lines] # 源句子转换成id列表
        target_in = [[utils.START_ID] + utils.sentence2id(sentence, self.target_word2id) for sentence in self.__target_lines] # 目标句子输入在句首+START_ID
        target_out = [utils.sentence2id(sentence, self.target_word2id) + [utils.END_ID] for sentence in self.__target_lines ] # 目标句子输出在句尾+END_ID
        return src, target_in, target_out

# 数据分片
def bacth_yield(data, batch_size, shuffle = False):
    if shuffle:
        random.shuffle(data)
    print()
    src, target_in, target_out = [], [], []
    for _src, _target_in, _target_out in data:
        if len(src) == batch_size:
            yield src, target_in, target_out
            src, target_in, target_out = [], [], []
        src.append(_src)
        target_in.append(_target_in)
        target_out.append(_target_out)
    if len(src) == batch_size:
        yield src, target_in, target_out

# 数据填充
def padding(src, target_in, target_out, pad_mark = 0):
    src_max_len = max(map(lambda x: len(x), src))
    target_max_len = max(map(lambda x: len(x), target_in))
    src_list, src_len_list = [], []

    target_in_list, target_out_list, target_len_list = [], [], []

    for _src, _target_in, _target_out in zip(src, target_in, target_out):
        src_ = _src[:src_max_len] + [pad_mark] * max(src_max_len - len(_src), 0)
        src_list.append(src_)
        src_len_list.append(min(len(_src), src_max_len))
    #for _target_in, _target_out in zip(target_in, target_out):
        target_in_ = _target_in[:target_max_len] + [pad_mark] * max(target_max_len - len(_target_in), 0)
        target_in_list.append(target_in_)
        target_out_ = _target_out[:target_max_len] + [pad_mark] * max(target_max_len - len(_target_out), 0)
        target_out_list.append(target_out_)
        target_len_list.append(min(len(_target_in), target_max_len))
    return src_list, src_len_list, target_in_list, target_out_list, target_len_list

if __name__ == '__main__':
    src_path = './data/train.zh.tok'
    target_path = './data/train.en.tok'
    dataloader = DataLoader(src_path, target_path)
    #print(dataloader.data[0])
    batches = bacth_yield(dataloader.data, 64)
    for i, (src, target_in, target_out) in enumerate(batches):
        print(i, len(src))
        #print(target_out)
    '''
    batches = bacth_yield(train_loader.data, 5)
    for i, (src, target_in, target_out) in enumerate(batches):
        #print(target_out)
        _,_,_,_,test = padding(src, target_in, target_out)
        print(test)
        break
    '''