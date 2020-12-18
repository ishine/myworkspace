#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import argparse
import os
import random
import json
import numpy as np


class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])


        # determine the model name and model dir
        if self.model_name is None:
            self.model_name = 'CRCNN'
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # backup data
        self.__config_backup(args)



    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # several key selective parameters
        parser.add_argument('--data_dir', type=str,
                            default='./data',
                            help='dir to load data')
        parser.add_argument('--output_dir', type=str,
                            default='./output',
                            help='dir to save output')

        # word embedding
        parser.add_argument('--embedding_path', type=str,
                            default='./embedding/glove.6B.300d.txt',
                            help='pre_trained word embedding')
        parser.add_argument('--word_dim', type=int,
                            default=300,
                            help='dimension of word embedding')

        # train settings
        parser.add_argument('--model_name', type=str,
                            default=None,
                            help='model name')
        parser.add_argument('--mode', type=int,
                            default=1,
                            choices=[0, 1],
                            help='running mode: 1 for training; otherwise testing')
        parser.add_argument('--epoch', type=int,
                            default=100,
                            help='max epoches during training')

        # hyper parameters
        parser.add_argument('--dropout', type=float,
                            default=0.5,
                            help='the possiblity of dropout')
        parser.add_argument('--batch_size', type=int,
                            default=128,
                            help='batch size')
        parser.add_argument('--lr', type=float,
                            default=0.01,
                            help='learning rate')
        parser.add_argument('--max_len', type=int,
                            default=100,
                            help='max length of sentence')
        parser.add_argument('--pos_dis', type=int,
                            default=50,
                            help='max distance of position embedding')
        parser.add_argument('--pos_dim', type=int,
                            default=70,
                            help='dimension of position embedding')

        # hyper parameters for cnn
        parser.add_argument('--filter_num', type=int,
                            default=1000,
                            help='the number of filters in convolution')
        parser.add_argument('--window', type=int,
                            default=3,
                            help='the size of window in convolution')

        parser.add_argument('--margin_positive', type=float,
                            default=2.5,
                            help='positive margin in the CRCNN loss function')
        parser.add_argument('--margin_negative', type=float,
                            default=0.5,
                            help='negative margin in the CRCNN loss function')
        parser.add_argument('--gamma', type=float,
                            default=2.0,
                            help='scaling factor `gamma` in the CRCNN loss function')
        parser.add_argument('--L2_decay', type=float,
                            default=1e-3,
                            help='L2 weight decay')
        args = parser.parse_args()
        return args


    def __config_backup(self, args):
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = Config()
    config.print_config()
