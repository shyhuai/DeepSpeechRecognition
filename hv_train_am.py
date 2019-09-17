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
import horovod.keras as hvd

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
data_args.batch_size = args.batch_size
data_args.data_length = None
data_args.shuffle = False
dev_data = get_data(data_args)

# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am_args.gpu_nums = 1
am_args.lr = args.lr
am_args.is_training = True
am = Am(am_args)
ctc_model = am.ctc_model

#opt = Adam(lr = args.lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
opt = SGD(lr=args.lr, momentum=0.9, clipnorm=400.0)
opt = hvd.DistributedOptimizer(opt)

ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)


#if os.path.exists('logs_am/model.h5'):
if args.pretrain is not None and os.path.exists(args.pretrain):
    logger.info('load acoustic model: %s ...' % args.pretrain)
    ctc_model.load_weights(args.pretrain)


epochs = args.epochs
batch_num = len(train_data.wav_lst) // train_data.batch_size // hvd.size()
batch_num_of_dev  = len(dev_data.wav_lst) // dev_data.batch_size
logger.info('# of samples: %d', len(train_data.wav_lst))
logger.info('mini-batch size: %d', train_data.batch_size)
logger.info('# of iterations per epoch: %d', batch_num)


callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        LossAndErrorPrintingCallback(batch_num, ctc_model), 
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
        ]

if hvd.rank() == 0:
    ckpt = "%s_model_{epoch:02d}.hdf5" % EXPERIMENT
    checkpoint = ModelCheckpoint(os.path.join(args.saved_dir, ckpt), monitor='val_loss', save_weights_only=True, verbose=2, save_best_only=True)
    callbacks.append(checkpoint)

batch = train_data.get_am_batch()
dev_batch = dev_data.get_am_batch()

#ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, verbose=0, callbacks=callbacks, workers=NDATATHREADS, use_multiprocessing=True, validation_data=dev_batch, validation_steps=batch_num_of_dev)
ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, verbose=0, callbacks=callbacks, workers=NDATATHREADS, use_multiprocessing=True)

if hvd.rank() == 0:
    am.ctc_model.save_weights('%s/%s_model.h5' % (args.saved_dir, EXPERIMENT))
