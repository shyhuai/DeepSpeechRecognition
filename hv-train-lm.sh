#!/usr/bin/env bash
#cd /home/shaohuais/repos/dsr;
nworkers=4
logprefix=hvtrainlm
datasets=thchs30,aishell,prime,stcmd
epochs=100
lr=0.0003
initial_epoch=0
data_dir=/home/comp/15485625/data/speech/sp2chs
saved_dir=/home/comp/15485625/checkpoints/hvtrain-lm
#data_dir=/datasets/shshi/speech/sp2chs
#saved_dir=/datasets/shshi/checkpoint-alldata
#data_dir=/tmp/shshi/sp2chs
PY=/home/comp/15485625/tf1.13.1/bin/python
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include em1 \
    $PY hv_train_lm.py --log ./logs/${logprefix}-train.log --batch_size 32 --nworkers $nworkers --datasets $datasets --logprefix $logprefix --data_dir ${data_dir} --saved_dir $saved_dir --epochs $epochs --initial_epoch $initial_epoch --lr $lr
