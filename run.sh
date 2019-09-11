nworkers=8
logprefix=tamultigpu
datasets=thchs30,aishell
epochs=100
#python train.py --log ${logprefix}-train.log --batch_size 64 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir /home/comp/15485625/data/speech/sp2chs --saved_dir ./checkpoint --epochs $epochs
python train.py --log ${logprefix}-train.log --batch_size 64 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir /datasets/shshi/speech/sp2chs --saved_dir /datasets/shshi/checkpoint2 --epochs $epochs
