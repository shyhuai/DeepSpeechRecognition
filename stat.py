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
logger.info('# of am vocal: %d', len(train_data.am_vocab))
logger.info('# of pinyin vocal: %d', len(train_data.pny_vocab))
logger.info('# of hanzi vocal: %d', len(train_data.han_vocab))
