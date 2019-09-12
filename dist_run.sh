nworkers=4
logprefix=test
datasets=thchs30
epochs=2
batch_size=32
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=~/tf1.13.1/bin/python
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include em1 \
    $PY dist_train.py --log ./logs/${logprefix}-train.log --batch_size $batch_size --nworkers 1 --datasets $datasets --logprefix $logprefix --data_dir /home/comp/15485625/data/speech/sp2chs --saved_dir ./checkpoint2 --epochs $epochs
