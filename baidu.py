#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

ak = 'H8BcNhFKMhmlwynyUB4YegOI'
ai = '10469665'
sk = '3272504460441b5bd0d8c43294418b7a'

import wave
from aip import AipSpeech
asp = AipSpeech(ai,ak,sk)

#DATAPATH='/media/sf_Shared_Data/tmp/drchu/dsr/'
#filename='output.wav'
#fn = os.path.join(DATAPATH, filename)
#fn = "/home/comp/15485625/speechrealtest/output2.wav"
fn = "/Users/lele/work/testdata/output.wav"
#fn = [
#        "./tmp/tmp0000.wav",
#        "./tmp/tmp0001.wav",
#        ]

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

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

max_len = 10
def split_audio_file(filename):
    if type(filename) is list:
        return filename
    testfile = wave.open(fn, mode='rb')
    framerate = testfile.getframerate()
    framenum = testfile.getnframes()
    length = framenum/framerate
    filelist = []

    if length > max_len:
        piece_len = max_len #(max_len // 3) * 2
        portion = piece_len * framerate
        n_pieces = length // piece_len + 1
        n_pieces = int(n_pieces)

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
    else:
        filelist.append(filename)
    testfile.close()
    return filelist

filelist = split_audio_file(fn)
for fn in filelist:
    r = asp.asr(get_file_content(fn), 'pcm', 16000, {
        'dev_pid': 1536,
    })
    #print(r)
    if r['err_no'] > 0:
        continue
    print('%s: %s' % (fn, r['result'][0]))

