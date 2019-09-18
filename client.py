import socket
from settings import ai,ak,sk
from aip import AipSpeech
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
while True:
    from_server = client.recv(4096)
    modified_message = from_server.decode('utf-8')
    ans = asp.synthesis(modified_message,'zh',1,{'vol':5,'per' : 2, 'pit' : 6,'spd' : 6,'cuid':123})
    if not isinstance(ans, dict):
        with open(voice_path + '/n' + str(audio_number) + 'audio.wav','wb') as f:
            f.write(ans)
    # print(from_server)
    print('Message  ' + str(audio_number) + '：' + modified_message + '\n')
    audio_number +=1

client.close()














