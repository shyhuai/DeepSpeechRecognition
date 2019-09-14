#!/usr/bin/env bash
cd /home/shaohuais/repos/dsr;
nworkers=8
#logprefix=thchs30-aishell
#datasets=thchs30,aishell
logprefix=thchs30-aishell-finetune-0.65
datasets=thchs30,aishell
epochs=199
#lr=0.0003
lr=0.000003
initial_epoch=198
#data_dir=/home/comp/15485625/data/speech/sp2chs
data_dir=/datasets/shshi/speech/sp2chs
saved_dir=/datasets/shshi/checkpoint-aishell-finetune-0.65
#data_dir=/tmp/shshi/sp2chs
PY=/home/shaohuais/tf1.13.1/bin/python
$PY train.py --log ./logs/${logprefix}-train.log --batch_size 128 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir ${data_dir} --saved_dir $saved_dir --epochs $epochs --initial_epoch $initial_epoch --lr $lr --pretrain /datasets/shshi/checkpoint-aishell/thchs30-aishell_model_197.hdf5 --training_models am
#python train.py --log ${logprefix}-train.log --batch_size 64 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir /datasets/shshi/speech/sp2chs --saved_dir /datasets/shshi/checkpoint2 --epochs $epochs
