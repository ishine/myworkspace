import os 
import tensorflow as tf 

from config import Config
from utils import load_embedding, load_relation, DataLoader


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')

    word2id, word_vec = load_embedding(config.embedding_path, config.word_dim)
    rel2id, id2rel, class_num = load_relation(config.data_dir)
    loader = DataLoader(rel2id, word2id, config)

    if config.mode == 1:
        train_loader = loader.get_train().repeat(config.epoch)
        dev_loader = loader.get_dev()
    test_loader = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    print('finish!')
    embedding = tf.Variable(tf.convert_to_tensor(word_vec))
    print(type(embedding))
    print(embedding.shape)

    
