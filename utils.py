import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from keras import backend as K

def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type='train',
        data_path='data/',
        thchs30=False,
        aishell=False,
        prime=False,
        stcmd=False,
        batch_size=1,
        data_length=None,
        shuffle=True)
    return params

def assign_datasets(datasets, param):
    for d in datasets:
        if d == 'thchs30':
            param.thchs30= True
        elif d == 'aishell':
            param.aishell = True
        elif d == 'prime':
            param.prime = True
        elif d == 'stcmd':
            param.stcmd = True


class get_data():
    def __init__(self, args):
        self.data_type = args.data_type
        self.data_path = args.data_path
        self.thchs30 = args.thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.source_init()

    def source_init(self):
        print('get source list...')
        read_files = []
        if self.data_type == 'train':
            if self.thchs30 == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.stcmd == True:
                read_files.append('stcmd.txt')
        elif self.data_type == 'dev':
            if self.thchs30 == True:
                read_files.append('thchs_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
        elif self.data_type == 'test':
            if self.thchs30 == True:
                read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
        self.wav_lst = []
        self.pny_lst = []
        self.han_lst = []
        for file in read_files:
            print('load ', file, ' data...')
            sub_file = 'data/' + file
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data):
                wav_file, pny, han = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                self.han_lst.append(han.strip('\n'))
        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]
        print('make am vocab...')
        if os.path.exists('./data/am_vocab.npy'):
            self.am_vocab = list(np.load('./data/am_vocab.npy'))
        else:
            self.am_vocab = self.mk_am_vocab(self.pny_lst)
            np.save('./data/am_vocab.npy', np.array(self.am_vocab))

        print('make lm pinyin vocab...')
        if os.path.exists('./data/pny_vocab.npy'):
            self.pny_vocab = list(np.load('./data/pny_vocab.npy'))
        else:
            self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)
            np.save('./data/pny_vocab.npy', np.array(self.pny_vocab))

        print('make lm hanzi vocab...')
        if os.path.exists('./data/han_vocab.npy'):
            self.han_vocab = list(np.load('./data/han_vocab.npy'))
        else:
            self.han_vocab = self.mk_lm_han_vocab(self.han_lst)
            np.save('./data/han_vocab.npy', np.array(self.han_vocab))

    def get_am_batch(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []
                label_data_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    fbank = compute_fbank(os.path.join(self.data_path, self.wav_lst[index]))
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    if pad_fbank.shape[0] // 8 >= label_ctc_len:
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length,
                          'label_length': label_length,
                          }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                yield inputs, outputs

    def get_dep_batch(self, wav_list):
        #shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            #if self.shuffle == True:
            #    shuffle(shuffle_list)
            wav_list.sort()
            #                                                           print(wav_list)
            for i in range(len(wav_list) // self.batch_size):
                wav_data_lst = []
                label_data_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = wav_list[begin:end]
                ############################################print(wav_list, sub_list)
                for index in sub_list:
                    #fbank = compute_fbank('./tmp/'+index)
                    fbank = compute_fbank(index)
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank
                    #label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    #label_ctc_len = self.ctc_len(label)
                    #if pad_fbank.shape[0] // 8 >= label_ctc_len:
                    #    wav_data_lst.append(pad_fbank)
                    #    label_data_lst.append(label)
                #if len(wav_data)
                #pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_wav_data = pad_fbank
                wav_lst = np.zeros((1, len(pad_wav_data), 200, 1))
                wav_lst[0, :len(pad_wav_data), :, 0]=pad_wav_data
                #wav_lst[i][:len(pad_wav_data)]=pad_wav_data
                #pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': wav_lst,
                          'the_labels': ["Not given"],
                          'input_length': 1,
                          'label_length': 1,
                          }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                yield inputs, outputs

    def get_lm_batch(self):
        batch_num = len(self.pny_lst) // self.batch_size
        shuffle_list = [i for i in range(len(self.pny_lst))]
        if self.shuffle == True:
            shuffle(shuffle_list)
        pny_lst = [self.pny_lst[i] for i in shuffle_list]
        han_lst = [self.han_lst[i] for i in shuffle_list] 
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            input_batch = pny_lst[begin:end]
            label_batch = han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [self.pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
                #[self.pny2id(''.join(line.split()), self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                #[self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
                [self.han2id(''.join(line.split()), self.han_vocab) + [0] * (max_len - len(''.join(line.split()))) for line in label_batch])
            yield input_batch, label_batch

    def pny2id(self, line, vocab):
        return [vocab.index(pny) for pny in line]

    def han2id(self, line, vocab):
        indices = []
        for han in line:
            try:
                i = vocab.index(han)
            except:
                print('Error: ', line, han)
                raise
            indices.append(i)
        return indices
        #return [vocab.index(han) for han in line]

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_am_vocab(self, data):
        vocab = []
        for line in tqdm(data):
            line = line
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    def mk_lm_pny_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab

    def mk_lm_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len


# 对音频文件提取mfcc特征
def compute_mfcc(file):
    fs, audio = wav.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
    mfcc_feat = mfcc_feat[::3]
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat


# 获取信号的时频图
def compute_fbank(file):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input


# word error rate------------------------------------
def GetEditDistance(str1, str2):
	leven_cost = 0
	s = difflib.SequenceMatcher(None, str1, str2)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		if tag == 'replace':
			leven_cost += max(i2-i1, j2-j1)
		elif tag == 'insert':
			leven_cost += (j2-j1)
		elif tag == 'delete':
			leven_cost += (i2-i1)
	return leven_cost

# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text

def create_path(relative_path):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, relative_path)
    if not os.path.isdir(filename):
        try:
            os.makedirs(filename)
        except:
            pass

def readfiles(txtfn):
    files = []
    with open(txtfn) as f:
        for l in f.readlines():
            files.append(l[0:-1])
    return files

