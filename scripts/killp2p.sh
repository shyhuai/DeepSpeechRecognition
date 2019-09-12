kill -9 `ps aux|grep 'python train.py' | awk '{print $2}'`
