#coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance
import wave

# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
data_args = data_hparams()
train_data = get_data(data_args)

from multiprocessing import Process
from pyaudio import PyAudio,paInt16

framerate=16000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=10

# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('logs_am/thchs30-aishell-finetune_model.h5')
#am.ctc_model.load_weights('logs_am/model.h5')

# 2.语言模型-------------------------------------------
from model_language.transformer import Lm, lm_hparams

lm_args = lm_hparams()
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm_args.dropout_rate = 0.
print('loading language model...')
lm = Lm(lm_args)
sess = tf.Session(graph=lm.graph)
with lm.graph.as_default():
    saver =tf.train.Saver()
with sess.as_default():
    latest = tf.train.latest_checkpoint('logs_lm')
    saver.restore(sess, latest)

# 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
#    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
#    且会出现未出现的词。
data_args.data_type = 'test'
data_args.shuffle = False
data_args.batch_size = 1
test_data = get_data(data_args)

# 4. 进行测试-------------------------------------------
def predict(thefile):
    if not thefile.endswith('wav'):
        print("[Error*] The file is not in .wav format!")
    testfile = wave.open('./tmp/'+thefile, mode='rb')
    print("The input has {} channel(s)".format(testfile.getnchannels()))
    am_batch = ''
    framerate = testfile.getframerate()
    framenum = testfile.getnframes()
    length = framenum/framerate
    print("The length of {} is {} seconds.".format(thefile, length))
    max_len = 10
    if length > max_len:
        piece_len = (max_len // 3) * 2
        portion = piece_len * framerate
        n_pieces = length // piece_len + 1
        n_pieces = int(n_pieces)
        print("The file exceeds the max length of {} seconds and needs to be split into {} pieces".format(max_len, n_pieces))
        for i in range(n_pieces):
            apiece = testfile.readframes(framerate*max_len)
            testfile.setpos(testfile.tell()-portion/2)
            tmp = wave.open('./tmp/tmp{:04}.wav'.format(i), mode='wb')
            tmp.setnchannels(1)
            tmp.setframerate(16000)
            tmp.setsampwidth(2)
            tmp.writeframes(apiece)
            tmp.close()
        am_batch = test_data.get_dep_batch(os.listdir('./tmp/'))
        for i in range(n_pieces):
            inputs, _ = next(am_batch)
            x = inputs['the_inputs']
            #print(x.shape)
            #y = test_data.pny_lst[i]
            result = am.model.predict(x, steps=1)
            # 将数字结果转化为文本结果
            _, text = decode_ctc(result, train_data.am_vocab)
            text = ' '.join(text)
            print('文本结果：', text)
            #print('原文结果：', ' '.join(y))
            with sess.as_default():
                text = text.strip('\n').split(' ')
                x = np.array([train_data.pny_vocab.index(pny) for pny in text])
                x = x.reshape(1, -1)
                preds = sess.run(lm.preds, {lm.x: x})
                #label = test_data.han_lst[i]
                got = ''.join(train_data.han_vocab[idx] for idx in preds[0])
                #print('原文汉字：', label)
                print('识别结果：', got)
                #word_error_num += min(len(label), GetEditDistance(label, got))

    else:
        thelist = [thefile]
        am_batch = test_data.get_dep_batch(thelist)
        inputs, _ = next(am_batch)
        x = inputs['the_inputs']
        #print(x.shape)
        #y = test_data.pny_lst[i]
        result = am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        _, text = decode_ctc(result, train_data.am_vocab)
        text = ' '.join(text)
        print('文本结果：', text)
        #print('原文结果：', ' '.join(y))
        with sess.as_default():
            text = text.strip('\n').split(' ')
            x = np.array([train_data.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            preds = sess.run(lm.preds, {lm.x: x})
            #label = test_data.han_lst[i]
            got = ''.join(train_data.han_vocab[idx] for idx in preds[0])
            #print('原文汉字：', label)
            print('识别结果：', got)
            #word_error_num += min(len(label), GetEditDistance(label, got))
            #word_num += len(label)

def save_wave_file(filename,data):
    '''save the date to the wavfile'''
    wf=wave.open("./tmp/" + filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()
    predict(filename)

def my_record():
    cnt = 0
    while 1:
        pa=PyAudio()
        stream=pa.open(format = paInt16,channels=1,
                       rate=framerate,input=True,
                       frames_per_buffer=NUM_SAMPLES)
        my_buf=[]
        count=0
        while count<TIME*8:#控制录音时间
            string_audio_data = stream.read(NUM_SAMPLES)
            my_buf.append(string_audio_data)
            count+=1
        save_wave_file('tmp{:04}.wav'.format(cnt), my_buf)
        #p = Process(target=save_wave_file, args=('tmp{:04}.wav'.format(cnt),my_buf),)
        #p.start()
        stream.close()
        cnt = cnt + 1

if __name__ == '__main__':

    my_record()
    print('Over!')

