import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import os

filepath = '../testdata/D8_993.wav'

fs, wavsignal = wav.read(filepath)
print('# of frames: ', len(wavsignal))

plt.xlabel('Frame')
plt.ylabel('Amplitude')
#plt.plot(wavsignal)
#plt.show()

time_window = 25
window_length = fs // 1000 * time_window
print('window_length: ', window_length)

# Construct a hamming window with 400 frames
x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
plt.xlabel('Frame')
#plt.plot(w)
#plt.show()

# Split the frames with the window length
p_begin = 0
p_end = p_begin + window_length
frame = wavsignal[p_begin:p_end]
plt.plot(frame)
plt.xlabel('Frame')
plt.ylabel('Amplitude')
plt.show()

# 加窗
frame = frame * w
plt.plot(frame)
plt.xlabel('Frame')
plt.ylabel('Amplitude')
plt.show()

from scipy.fftpack import fft

# 进行快速傅里叶变换
frame_fft = np.abs(fft(frame))[:200]
plt.plot(frame_fft)
plt.show()

# 取对数，求db
frame_log = np.log(frame_fft)
plt.plot(frame_log)
plt.show()

# 获取信号的时频图
def compute_fbank(file):
    x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
    fs, wavsignal = wav.read(file)
# wav波形 加时间窗以及时移10ms
    time_window = 25 # 单位ms
    window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
    wav_arr = np.array(wavsignal)
    wav_length = len(wavsignal)
    range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype = np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]  
        data_line = data_line * w # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
#data_input = data_input[::]
    return data_input


a = compute_fbank(filepath)
plt.imshow(a.T, origin = 'lower')
plt.show()
