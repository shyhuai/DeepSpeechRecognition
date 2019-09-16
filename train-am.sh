#!/usr/bin/env bash
cd /home/shaohuais/repos/dsr;
nworkers=8
#logprefix=thchs30-aishell
#datasets=thchs30,aishell
logprefix=alldata
datasets=thchs30,aishell,prime,stcmd
epochs=200
lr=0.0003
initial_epoch=84
#data_dir=/home/comp/15485625/data/speech/sp2chs
data_dir=/datasets/shshi/speech/sp2chs
saved_dir=/datasets/shshi/checkpoint-alldata
#data_dir=/tmp/shshi/sp2chs
python train.py --log ./logs/${logprefix}-train.log --batch_size 128 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir ${data_dir} --saved_dir $saved_dir --epochs $epochs --initial_epoch $initial_epoch --lr $lr --training_models am --pretrain /datasets/shshi/checkpoint-alldata/alldata_model_83.hdf5
#python train.py --log ${logprefix}-train.log --batch_size 64 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir /datasets/shshi/speech/sp2chs --saved_dir /datasets/shshi/checkpoint2 --epochs $epochs
