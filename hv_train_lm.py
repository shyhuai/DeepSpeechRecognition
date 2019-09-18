import os
import argparse, os
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras import backend as K
import keras
from utils import get_data, data_hparams, assign_datasets, create_path
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
import logging
from settings import logger, formatter
from callbacks import LossAndErrorPrintingCallback, lr_scheduler
import horovod.tensorflow as hvd
import time

if True:
    parser = argparse.ArgumentParser(description="Distributed acoustic model trainer")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log', type=str, default='train.log')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Set start epoch')
    parser.add_argument('--nworkers', type=int, default=4, help='# of GPUs')
    parser.add_argument('--datasets', type=str, default='aishell', help='Specify the dataset for training: thchs30,aishell,prime,stcmd')
    parser.add_argument('--logprefix', type=str, default='exp1', help='Specify the log prefix')
    parser.add_argument('--data_dir', type=str, default='/tmp/shshi/sp2chs/', help='Specify the data root path')
    parser.add_argument('--pretrain', type=str, default=None, help='Load pretrain model')
    parser.add_argument('--saved_dir', type=str, default='/tmp/shshi/checkpoint', help='Specify the saved weights or gradients root path')
    parser.add_argument('--lr', type=float, default=0.0008, help='Default learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Default maximum epochs to train')
    args = parser.parse_args()

create_path(args.saved_dir)

hvd.init()
os.environ["CUDA_VISIBLE_DEVICES"]=str(hvd.rank())
logfile = args.log.split('.log')[0]+str(hvd.rank())+'.log'
hdlr = logging.FileHandler(logfile)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.info('Configurations: %s', args)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


EXPERIMENT=args.logprefix
NWORKERS=args.nworkers
NDATATHREADS=4


datasets = args.datasets.split(',')
# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'train'
data_args.data_path = args.data_dir
assign_datasets(datasets, data_args)
data_args.batch_size = args.batch_size
data_args.data_length = None
data_args.shuffle = True
train_data = get_data(data_args)

# 0.准备验证所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'dev'
data_args.data_path = args.data_dir
assign_datasets(datasets, data_args)
data_args.batch_size = args.batch_size * hvd.size()
data_args.data_length = None
data_args.shuffle = False
dev_data = get_data(data_args)

epochs=args.epochs
batch_num = len(train_data.wav_lst) // train_data.batch_size
batch_num_of_dev  = len(dev_data.wav_lst) // dev_data.batch_size

if True:
    from model_language.transformer import Lm, lm_hparams
    lm_args = lm_hparams()
    lm_args.num_heads = 8
    lm_args.num_blocks = 6
    lm_args.input_vocab_size = len(train_data.pny_vocab)
    lm_args.label_vocab_size = len(train_data.han_vocab)
    logger.info('# of pinyin vaocal: %d', lm_args.input_vocab_size)
    logger.info('# of hanzi vaocal: %d', lm_args.label_vocab_size)
    lm_args.max_length = 100
    lm_args.hidden_units = 512
    lm_args.dropout_rate = 0.2
    lm_args.lr = 0.0003
    lm_args.nworkers = args.nworkers
    lm_args.hvd = hvd

    lm_args.is_training = True
    lm = Lm(lm_args)
    rank = hvd.rank()
    local_bs = args.batch_size
    hooks = [hvd.BroadcastGlobalVariablesHook(0),
            tf.train.StopAtStepHook(last_step=batch_num * epochs),
            ]
    
    with lm.graph.as_default():
        saver =tf.train.Saver()
        merged = tf.summary.merge_all()
        #global_init = tf.global_variables_initializer()
        with tf.train.MonitoredTrainingSession(hooks=hooks, config=config) as sess:
            #sess.run(global_init)
            add_num = 0
            for k in range(epochs):
                total_loss = 0
                batch = train_data.get_lm_batch()
                stime = time.time()
                avg_time = 0.0
                display = 400
                tmp_loss = 0.0
                for i in range(batch_num):
                    input_batch, label_batch = next(batch)
                    #input_batch, label_batch = input_batch[rank*local_bs:(rank+1)*local_bs],label_batch[rank*local_bs:(rank+1)*local_bs] 
                    feed = {lm.x: input_batch, lm.y: label_batch}
                    cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
                    total_loss += cost
                    tmp_loss += cost
                    avg_time += time.time() - stime

                    if i > 0 and i % display == 0:
                        logger.info('[rank:%d] epoch: %d [%d/%d] loss: %f, time used per iteration: %f', rank, k, i, batch_num, tmp_loss/display, avg_time/display)
                        tmp_loss = 0.0
                        stime = time.time()
                        avg_time = 0.0
                logger.info('[rank:%d] epochs: %d%s%f', rank, k+1, ': average loss = ', total_loss/batch_num)

                # Evaluation
                if rank == 0:
                    dev_batch = dev_data.get_lm_batch()
                    val_loss = 0.0
                    for i in range(batch_num_of_dev):
                        input_batch, label_batch = next(dev_batch)
                        loss = sess.run(lm.mean_loss, feed_dict=feed)
                        val_loss += loss
                    logger.info('[rank:%d] epochs: %d%s%f', rank, k+1, ': val loss = ', val_loss/batch_num_of_dev)
                    if k > 0 and k % 5 == 0:
                        saver.save(sess, '%s/lm%s_model_%d' % (args.saved_dir, EXPERIMENT, k))
            if rank == 0:
                saver.save(sess, '%s/lm%s_model_%d' % (args.saved_dir, EXPERIMENT, epochs))
            #writer.close()
