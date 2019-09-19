# -*- coding: utf-8 -*-
from __future__ import print_function
import logging
from settings import logger

from utils import get_data, data_hparams, assign_datasets, create_path

DATASETS='thchs30,aishell,prime,stcmd'
data_dir='/home/comp/15485625/data/speech/sp2chs'
datasets = DATASETS.split(',')

# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'train'
data_args.data_path = data_dir
assign_datasets(datasets, data_args)
data_args.batch_size = 32 
data_args.data_length = None
data_args.shuffle = False
train_data = get_data(data_args)

logger.info('# of samples: %d', len(train_data.wav_lst))
logger.info('# of pny samples: %d', len(train_data.pny_lst))
logger.info('# of hanzi samples: %d', len(train_data.han_lst))
logger.info('# of am vocal: %d', len(train_data.am_vocab))
logger.info('# of pinyin vocal: %d', len(train_data.pny_vocab))
logger.info('# of hanzi vocal: %d', len(train_data.han_vocab))
#logger.info('hanzi vocal: %s', train_data.han_vocab)

pinyin_freq = {}
hanzi_freq = {}

def stat_freq(double_lst, freq_dict):
    idx = 0
    for lst in double_lst:
        for i in lst:
            if not (i in freq_dict):
                freq_dict[i] = idx
                idx += 1
            #freq_dict[i] += 1
            #freq_dict[i] = 1
stat_freq(train_data.pny_lst, pinyin_freq)
stat_freq(train_data.han_lst, hanzi_freq)

logger.info('pny item: %s', train_data.pny_lst[10])
print([pinyin_freq[i] for i in train_data.pny_lst[10]])
logger.info('hanzi item: %s', train_data.han_lst[10])
print([hanzi_freq[i] for i in train_data.han_lst[10]])
#print(pinyin_freq)
#print(hanzi_freq)
#logger.info('pingyin freq: %s', pinyin_freq)
#logger.info('hanzi freq: %s', hanzi_freq)
