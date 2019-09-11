nworkers=4
logprefix=tamultigpu
datasets=thchs30,aishell
epochs=100
python train.py --log ${logprefix}-train.log --batch_size 32 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir /home/comp/15485625/data/speech/sp2chs --saved_dir ./checkpoint --epochs $epochs
