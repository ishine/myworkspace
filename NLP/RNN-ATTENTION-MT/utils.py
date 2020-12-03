import logging
from six.moves import cPickle

UNK_ID = 0
START_ID = 1
END_ID = 2
PAD_ID = 3

UNK = '<UNK>'
START = '<START>'
END = '<END>'
PAD = 'PADDING'


def get_logger(logname):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def pickle_dump(filename, data):
    with open(filename, 'wb') as f:
        cPickle.dump(data, f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data
# UNK_ID = 0
def sentence2id(sentence, word2id):
    global UNK_ID
    return [word2id.get(w, UNK_ID) for w in sentence]
# UNK = 'UNK'
def id2sentence(idlist, id2word):
    global UNK
    return [id2word.get(id, UNK) for id in idlist]

