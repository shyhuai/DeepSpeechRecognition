import socket
from settings import ai,ak,sk
from aip import AipSpeech
import simpleaudio as sa
import librosa
import soundfile as sf

server_IP = '127.0.0.1'
server_PORT = 15678
asp = AipSpeech(ai,ak,sk)
voice_path = './client_voices'
# server_IP = '0.0.0.0'
# server_PORT = 15678
message = 'I am Client\n'
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((server_IP, server_PORT))
client.send(message.encode('utf-8'))
audio_number = 1

def play_sound(fn):
    x,_ = librosa.load(fn, sr=16000)
    tmpfn = './tmp/tmp.wav'
    sf.write(tmpfn, x, 16000)
    wave_obj = sa.WaveObject.from_wave_file(tmpfn)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing

cache = ''
while True:
    from_server = client.recv(1024)
    msg = from_server.decode('utf-8')
    cache += msg
    try:
        index = cache.index(':')
    except:
        continue
    modified_message = cache[0:index]
    cache = cache[index+1:]

    print('Message  ' + str(audio_number) + '：' + modified_message + '\n')
    ans = asp.synthesis(modified_message,'zh',1,{'vol':5,'per' : 1, 'pit' : 6,'spd' : 5,'cuid':123})
    if not isinstance(ans, dict):
        fn = voice_path + '/n' + str(audio_number) + 'audio.wav'
        with open(fn,'wb') as f:
            f.write(ans)
        play_sound(fn)
    # print(from_server)
    audio_number +=1

client.close()














