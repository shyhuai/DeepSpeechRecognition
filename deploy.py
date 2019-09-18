#coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance, assign_datasets
import wave
import socket
import argparse
from multiprocessing import Process
from pyaudio import PyAudio,paInt16
server_IP = '127.0.0.1'
server_PORT = 15678

framerate=16000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=10

# send_message = '我是服务器'
# content2='浸会大学异构计算实验室深度学习评测系统'
# content3='浸会大学异构计算实验室'
# content8='撑着油纸伞，独自。彷徨在悠长、悠长。又寂寥的雨巷。我希望逢着。一个丁香一样地。结着愁怨的姑娘。她是有。丁香一样的颜色。丁香一样的芬芳。丁香一样的忧愁。在雨中哀怨。哀怨又彷徨。她彷徨在这寂寥的雨巷。撑着油纸伞。像我一样。像我一样地。默默彳亍着。冷漠、凄清，又惆怅。她默默地走近。走近，又投出。太息一般的眼光。她飘过。像梦一般地。像梦一般地凄婉迷茫。像梦中飘过。一枝丁香地。我身旁飘过这女郎。她静默地远了、远了。到了颓圮的篱墙。走尽这雨巷。在雨的哀曲里。消了她的颜色。散了她的芬芳。消散了，甚至她的。太息般的眼光。丁香般的惆怅。撑着油纸伞，独自。彷徨在悠长、悠长。又寂寥的雨巷。我希望飘过。一个丁香一样地。结着愁怨的姑娘。'
# serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# serv.bind((server_IP, server_PORT))
# serv.listen(5)
# while True:
#     print("wait connect")
#     conn, addr = serv.accept()
#     from_client = ''
#     while True:
#         data = conn.recv(4096)
#         # modified_RecvMessage, serverAddress = data.recvfrom(data.decode('utf-8'))
#         modified_RecvMessage = data.decode('utf-8')
#         if not data: break
#         from_client += modified_RecvMessage
#         print(from_client)
#         conn.send(send_message.encode('utf-8'))
#     conn.close()

data_dir='/home/comp/15485625/data/speech/sp2chs'
# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
DATASETS='thchs30,aishell,prime,stcmd'
am_trained_model='C:/Users/zhtang/Desktop/models/shshi/checkpoint-alldata/alldata_model.h5'
lm_trained_model='C:/Users/zhtang/Desktop/models/shshi/checkpoint-alldata-lm'

#DATASETS='thchs30,aishell'
#am_trained_model='/home/comp/15485625/checkpoints/checkpoint-aishell-finetune-6.24/thchs30-aishell-finetune2_model.h5'
#lm_trained_model='/home/comp/15485625/checkpoints/checkpoint-aishell-finetune-6.24'
#fn = "/home/comp/15485625/speechrealtest/leletest2.wav"
#fn = "/home/comp/15485625/speechrealtest/output4.wav"
fn = "/home/comp/15485625/speechrealtest/D8_993.wav"
thefile = fn


datasets = DATASETS.split(',')
from utils import get_data, data_hparams
data_args = data_hparams()
data_args.data_type = 'train'
assign_datasets(datasets, data_args)
#data_args.thchs30 = True
#data_args.aishell = True
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
def predict(thefile):
    if not thefile.endswith('wav'):
        print("[Error*] The file is not in .wav format!")
    testfile = wave.open(thefile, mode='rb')
    print("The input has {} channel(s)".format(testfile.getnchannels()))
    am_batch = ''
    framerate = testfile.getframerate()
    framenum = testfile.getnframes()
    length = framenum/framerate
    print("The length of {} is {} seconds.".format(thefile, length))
    max_len = 10
    if length > max_len:
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
                modified_RecvMessage = data.decode('utf-8')
                print(modified_RecvMessage)
                conn.send(got.encode('utf-8'))
                print('%s: %s' % (filelist[i], got))
                #word_error_num += min(len(label), GetEditDistance(label, got))
                #word_num += len(label)

    else:
        filelist = [thefile]
        am_batch = test_data.get_dep_batch(filelist)
        inputs, _ = next(am_batch)
        x = inputs['the_inputs']
        #print(x.shape)
        #y = test_data.pny_lst[i]
        result = am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        _, text = decode_ctc(result, train_data.am_vocab)
        text = ' '.join(text)
        print('%s: %s' % (filelist[0], text))
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
            modified_RecvMessage = data.decode('utf-8')
            print(modified_RecvMessage)
            conn.send(got.encode('utf-8'))
            print('%s: %s' % (filelist[0], got))
            #word_error_num += min(len(label), GetEditDistance(label, got))
            #word_num += len(label)
    # conn.send('end11112222'.decode('utf-8'))

def save_wave_file(filename,data):
    '''save the date to the wavfile'''
    wf=wave.open("./tmp/" + filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()
    predict("./tmp/" + filename)

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

    parser = argparse.ArgumentParser(description="c:1--default, 2--recording")
    parser.add_argument('--choose', type=int, default=1)
    args = parser.parse_args()

    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind((server_IP, server_PORT))
    serv.listen(5)
    print("waiting connect=====================================\n")
    conn, addr = serv.accept()
    data = conn.recv(4096)
    if args.choose == 1:
        predict(thefile)
    elif args.choose ==2:
        my_record()

    conn.close()
    sess.close()





