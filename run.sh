nworkers=8
logprefix=thchs30-aishell
datasets=thchs30,aishell
#logprefix=thchs30-2
#datasets=thchs30
epochs=2
lr=0.0002
initial_epoch=0
data_dir=/home/comp/15485625/data/speech/sp2chs
saved_dir=/datasets/shshi/checkpoint3
#data_dir=/tmp/shshi/sp2chs
python train.py --log ${logprefix}-train.log --batch_size 64 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir ${data_dir} --saved_dir $saved_dir --epochs $epochs --initial_epoch $initial_epoch --lr $lr
#--pretrain ./checkpoint2/thchs30-2_model_56.hdf5
#python train.py --log ${logprefix}-train.log --batch_size 64 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir /datasets/shshi/speech/sp2chs --saved_dir /datasets/shshi/checkpoint2 --epochs $epochs
