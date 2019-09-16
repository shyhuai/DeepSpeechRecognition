#coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance, assign_datasets
from keras.models import model_from_json

AM_EXPERIMENT='thchs30-aishell-finetune2'
LM_EXPERIMENT='lmthchs30-aishell-finetune-0.65'
DATASETS='thchs30,aishell'

#GPUHOME
#saved_dir='/home/comp/15485625/checkpoits/checkpoint-aishell-finetune-0.65'
#data_dir='/home/comp/15485625/data/speech/sp2chs'

#NVIDIA GPU
saved_dir='/datasets/shshi/checkpoint-aishell-finetune-6.24'
#saved_dir='/datasets/shshi/checkpoint-aishell'
data_dir='/datasets/shshi/speech/sp2chs'

#EXPERIMENT='thchs30-aishell'
#DATASETS='thchs30,aishell'
#saved_dir='/datasets/shshi/checkpoint-aishell'
#data_dir='/datasets/shshi/speech/sp2chs'

#EXPERIMENT='thchs30multigpu'

# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
datasets = DATASETS.split(',')
from utils import get_data, data_hparams
data_args = data_hparams()
data_args.data_path = data_dir 
data_args.data_length = None
assign_datasets(datasets, data_args)
train_data = get_data(data_args)


# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_args = am_hparams()
am_args.gpu_nums = 1
am_args.is_training = True
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('%s/%s_model.h5'%(saved_dir, AM_EXPERIMENT))
#am.ctc_model.load_weights('checkpoint2/%s_model_56.hdf5'%EXPERIMENT)

# 2.语言模型-------------------------------------------
from model_language.transformer import Lm, lm_hparams

lm_args = lm_hparams()
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)

print('Pinyin vocab size: %d' % lm_args.input_vocab_size)
print('Hanzi vocab size: %d' % lm_args.label_vocab_size)
lm_args.dropout_rate = 0.
print('loading language model...')
lm = Lm(lm_args)
sess = tf.Session(graph=lm.graph)
with lm.graph.as_default():
    saver =tf.train.Saver()
with sess.as_default():
    #latest = tf.train.latest_checkpoint('logs_lm')
    latest = tf.train.latest_checkpoint(saved_dir)
    saver.restore(sess, latest)

# 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
#    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
#    且会出现未出现的词。
data_args.data_type = 'test'
data_args.shuffle = False
data_args.batch_size = 1
test_data = get_data(data_args)
nsampels = len(test_data.wav_lst)
nsampels = len(test_data.wav_lst)

# 4. 进行测试-------------------------------------------
am_batch = test_data.get_am_batch()
word_num = 0
word_error_num = 0
for i in range(nsampels):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inputs, _ = next(am_batch)
    x = inputs['the_inputs']
    y = test_data.pny_lst[i]
    result = am.model.predict(x, steps=1)
    # 将数字结果转化为文本结果
    _, text = decode_ctc(result, train_data.am_vocab)
    text = ' '.join(text)
    print('文本结果：', text)
    print('原文结果：', ' '.join(y))
    with sess.as_default():
        text = text.strip('\n').split(' ')
        x = np.array([train_data.pny_vocab.index(pny) for pny in text])
        x = x.reshape(1, -1)
        preds = sess.run(lm.preds, {lm.x: x})
        label = test_data.han_lst[i]
        got = ''.join(train_data.han_vocab[idx] for idx in preds[0])
        print('原文汉字：', label)
        print('识别结果：', got)
        word_error_num += min(len(label), GetEditDistance(label, got))
        word_num += len(label)
print('词错误率：', word_error_num / word_num)
sess.close()
