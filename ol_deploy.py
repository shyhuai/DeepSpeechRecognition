#coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import difflib
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import argparse
import utils
from utils import decode_ctc, GetEditDistance, assign_datasets, readfiles
import wave

parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument('--fn', type=str, default=None)
args = parser.parse_args()
if args.fn is not None:
    fn = args.fn
    if fn.endswith('.txt'):
        fn = readfiles(fn)
    else:
        fn = fn.split(',')
else:
    fn = "/home/comp/15485625/speechrealtest/D8_993.wav"

data_dir='/home/comp/15485625/data/speech/sp2chs'
# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
DATASETS='thchs30,aishell,prime,stcmd'
am_trained_model='/Users/lele/work/models/checkpoint-alldata/alldata_model.h5'
lm_trained_model='/Users/lele/work/models/checkpoint-alldata-lm'
#am_trained_model='/home/comp/15485625/checkpoints/checkpoint-alldata/alldata_model.h5'
#lm_trained_model='/home/comp/15485625/checkpoints/checkpoint-alldata-lm'

#DATASETS='thchs30,aishell'
#am_trained_model='/home/comp/15485625/checkpoints/checkpoint-aishell-finetune-6.24/thchs30-aishell-finetune2_model.h5'
#lm_trained_model='/home/comp/15485625/checkpoints/checkpoint-aishell-finetune-6.24'
#fn = "/home/comp/15485625/speechrealtest/leletest2.wav"

#fn = [
#      "/Users/lele/work/testdata/D8_993.wav",
#      "/Users/lele/work/testdata/D8_994.wav",
#      "/Users/lele/work/testdata/D8_995.wav",
#      "/Users/lele/work/testdata/D8_996.wav", 
#      "/Users/lele/work/testdata/D8_997.wav",
#      "/Users/lele/work/testdata/D8_998.wav",
#      "/Users/lele/work/testdata/BAC009S0766W0140.wav",
#      "/Users/lele/work/testdata/BAC009S0766W0141.wav",
#      "/Users/lele/work/testdata/BAC009S0766W0142.wav",
#      "/Users/lele/work/testdata/BAC009S0766W0143.wav",
#      "/Users/lele/work/testdata/BAC009S0766W0144.wav",
#     ]
#fn = [
#      "/home/comp/15485625/speechrealtest/D8_993.wav",
#      "/home/comp/15485625/speechrealtest/D8_994.wav",
#      "/home/comp/15485625/speechrealtest/D8_995.wav",
#      "/home/comp/15485625/speechrealtest/D8_996.wav", 
#      "/home/comp/15485625/speechrealtest/D8_997.wav",
#      "/home/comp/15485625/speechrealtest/D8_998.wav",
#      "/home/comp/15485625/speechrealtest/BAC009S0766W0140.wav",
#      "/home/comp/15485625/speechrealtest/BAC009S0766W0141.wav",
#      "/home/comp/15485625/speechrealtest/BAC009S0766W0142.wav",
#      "/home/comp/15485625/speechrealtest/BAC009S0766W0143.wav",
#      "/home/comp/15485625/speechrealtest/BAC009S0766W0144.wav",
#     ]

#fn = "/home/comp/15485625/speechrealtest/D8_995.wav"
thefile = fn


datasets = DATASETS.split(',')
from utils import get_data, data_hparams
data_args = data_hparams()
data_args.data_type = 'train'
assign_datasets(datasets, data_args)
data_args.data_path = data_dir 
data_args.data_length = None
train_data = get_data(data_args)


# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am_args.gpu_nums = 1
am_args.is_training = True
am = Am(am_args)
print('loading acoustic model...')
#am.ctc_model.load_weights('/home/comp/15485625/checkpoints/checkpoint-aishell-finetune-0.65/thchs30-aishell-finetune-0.65_model.h5')
am.ctc_model.load_weights(am_trained_model)
#am.ctc_model.load_weights('/home/comp/15485625/checkpoints/checkpoint-aishell-finetune-6.24/thchs30-aishell-finetune2_model.h5')
#am.ctc_model.load_weights('logs_am/model.h5')

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
    #latest = tf.train.latest_checkpoint('/home/comp/15485625/checkpoints/checkpoint-aishell-finetune-6.24')
    latest = tf.train.latest_checkpoint(lm_trained_model)
    saver.restore(sess, latest)

# 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
#    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
#    且会出现未出现的词。
data_args.data_type = 'test'
data_args.shuffle = False
data_args.batch_size = 1
test_data = get_data(data_args)

# 4. 进行测试-------------------------------------------
#thefile = "/home/comp/15485625/chuyi_sunjiaman_mono.wav"
#thefile = "/home/comp/15485625/speechrealtest/leletest2.wav"
#thefile = "/home/comp/15485625/data/speech/sp2chs/data_aishell/wav/test/S0765/BAC009S0765W0121.wav"
pretestfile = thefile
if type(thefile) is list:
    pretestfile = thefile[0]
    #print('filelist: ', thefile)
if not pretestfile.endswith('wav'):
    print("[Error*] The file is not in .wav format!")

testfile = wave.open(pretestfile, mode='rb')
print("The input has {} channel(s)".format(testfile.getnchannels()))
am_batch = ''
framerate = testfile.getframerate()
framenum = testfile.getnframes()
length = framenum/framerate
#print("The length of {} is {} seconds.".format(thefile, length))
max_len = 10
if type(thefile) is not list and length > max_len:
    piece_len = max_len #(max_len // 3) * 2
    portion = piece_len * framerate
    n_pieces = length // piece_len + 1
    n_pieces = int(n_pieces)
    print("The file exceeds the max length of {} seconds and needs to be split into {} pieces".format(max_len, n_pieces))
    filelist = []
    for i in range(n_pieces):
        apiece = testfile.readframes(framerate*max_len)
        #testfile.setpos(testfile.tell()-portion)
        filename = './tmp/tmp{:04}.wav'.format(i)
        tmp = wave.open(filename, mode='wb')
        tmp.setnchannels(1)
        tmp.setframerate(16000)
        tmp.setsampwidth(2)
        tmp.writeframes(apiece)
        tmp.close()
        filelist.append(filename)
    #am_batch = test_data.get_dep_batch(os.listdir('./tmp/'))
    am_batch = test_data.get_dep_batch(filelist)
    #am_batch = test_data.get_dep_batch(os.listdir('/home/comp/15485625/data/speech/sp2chs/data_aishell/wav/test/'))
    for i in range(n_pieces):
        inputs, _ = next(am_batch)
        x = inputs['the_inputs']
        #print(x.shape)
        #y = test_data.pny_lst[i]
        result = am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        _, text = decode_ctc(result, train_data.am_vocab)
        text = ' '.join(text)
        print('%s: %s' % (filelist[i], text))
        #print('原文结果：', ' '.join(y))
        with sess.as_default():
            text = text.strip('\n').split(' ')
            x = np.array([train_data.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            preds = sess.run(lm.preds, {lm.x: x})
            #label = test_data.han_lst[i]
            got = ''.join(train_data.han_vocab[idx] for idx in preds[0])
            #print('原文汉字：', label)
            #print('识别结果：', got)
            print('%s: %s' % (filelist[i], got))
            #word_error_num += min(len(label), GetEditDistance(label, got))
            #word_num += len(label)

else:
    if type(thefile) is list:
        filelist = thefile
    else:
        filelist = [thefile]
    am_batch = test_data.get_dep_batch(filelist)
    for i in range(len(filelist)):
        inputs, _ = next(am_batch)
        x = inputs['the_inputs']
        #print(x.shape)
        #y = test_data.pny_lst[i]
        result = am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        _, text = decode_ctc(result, train_data.am_vocab)
        text = ' '.join(text)
        print('%s: %s' % (filelist[i], text))
        #print('原文结果：', ' '.join(y))
        with sess.as_default():
            text = text.strip('\n').split(' ')
            x = np.array([train_data.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            preds = sess.run(lm.preds, {lm.x: x})
            #label = test_data.han_lst[i]
            got = ''.join(train_data.han_vocab[idx] for idx in preds[0])
            #print('原文汉字：', label)
            #print('识别结果：', got)
            print('%s: %s' % (filelist[i], got))
            #word_error_num += min(len(label), GetEditDistance(label, got))
            #word_num += len(label)

sess.close()
