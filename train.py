# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import tensorflow as tf
import keras
from keras.optimizers import Adam, SGD
from utils import get_data, data_hparams, assign_datasets, create_path
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
import logging
from settings import logger, formatter
from callbacks import LossAndErrorPrintingCallback, lr_scheduler

if True:
    parser = argparse.ArgumentParser(description="Acoustic model trainer")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log', type=str, default='train.log')
    parser.add_argument('--nworkers', type=int, default=4, help='# of GPUs')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Set start epoch')
    parser.add_argument('--datasets', type=str, default='aishell', help='Specify the dataset for training: thchs30,aishell,prime,stcmd')
    parser.add_argument('--logprefix', type=str, default='exp1', help='Specify the log prefix')
    parser.add_argument('--data_dir', type=str, default='/tmp/shshi/sp2chs/', help='Specify the data root path')
    parser.add_argument('--pretrain', type=str, default=None, help='Load pretrain model')
    parser.add_argument('--saved_dir', type=str, default='/tmp/shshi/checkpoint', help='Specify the saved weights or gradients root path')
    parser.add_argument('--lr', type=float, default=0.0008, help='Default learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Default maximum epochs to train')
    args = parser.parse_args()

hdlr = logging.FileHandler(args.log)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.info('Configurations: %s', args)

EXPERIMENT=args.logprefix
NWORKERS=args.nworkers
NDATATHREADS=32


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
am_args.gpu_nums = NWORKERS
am_args.lr = args.lr
am_args.is_training = True
am = Am(am_args)
ctc_model = am.ctc_model

#opt = Adam(lr = args.lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8, clipnorm=400.0)
opt = SGD(lr=args.lr, momentum=0.9, clipnorm=400.0)

if args.nworkers > 1:
    ctc_model = multi_gpu_model(ctc_model, gpus=args.nworkers)

ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)


if args.pretrain is not None and os.path.exists(args.pretrain):
    logger.info('load acoustic model: %s ...' % args.pretrain)
    ctc_model.load_weights(args.pretrain)


epochs = args.epochs
batch_num = len(train_data.wav_lst) // train_data.batch_size
batch_num_of_dev  = len(dev_data.wav_lst) // dev_data.batch_size
logger.info('# of samples: %d', len(train_data.wav_lst))
logger.info('mini-batch size: %d', train_data.batch_size)
logger.info('# of iterations per epoch: %d', batch_num)

create_path(args.saved_dir)

# checkpoint
#ckpt = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
#ckpt = "model_{epoch:02d}.hdf5"
#ckpt = "primeonly_model_{epoch:02d}.hdf5"
ckpt = "%s_model_{epoch:02d}.hdf5" % EXPERIMENT
checkpoint = ModelCheckpoint(os.path.join(args.saved_dir, ckpt), monitor='val_loss', save_weights_only=True, verbose=2, save_best_only=False)
callbacks = [
        checkpoint, 
        LossAndErrorPrintingCallback(batch_num, am.ctc_model), 
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
        ]

batch = train_data.get_am_batch()
dev_batch = dev_data.get_am_batch()

#ctc_model.fit_generator(batch, initial_epoch=args.initial_epoch, steps_per_epoch=batch_num, epochs=epochs, verbose=0, callbacks=callbacks, workers=NDATATHREADS, use_multiprocessing=True, validation_data=dev_batch, validation_steps=batch_num_of_dev)
ctc_model.fit_generator(batch, initial_epoch=args.initial_epoch, steps_per_epoch=batch_num, epochs=epochs, verbose=0, callbacks=callbacks, workers=NDATATHREADS, use_multiprocessing=True)

am.ctc_model.save_weights('%s/%s_model.h5' % (args.saved_dir, EXPERIMENT))


# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams
lm_args = lm_hparams()
lm_args.num_heads = 8
lm_args.num_blocks = 6
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm_args.max_length = 100
lm_args.hidden_units = 512
lm_args.dropout_rate = 0.2
lm_args.lr = 0.0003
lm_args.is_training = True
lm = Lm(lm_args)
epochs=10

with lm.graph.as_default():
    saver =tf.train.Saver()
with tf.Session(graph=lm.graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0
    if os.path.exists('%s/checkpoint'%args.saved_dir):
        print('loading language model...')
        latest = tf.train.latest_checkpoint(args.saved_dir)
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                #writer.add_summary(rs, k * batch_num + i)
        logger.info('epochs: %d%s%f', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, '%s/%s_model_%d' % (args.saved_dir, EXPERIMENT, epochs + add_num))
    #writer.close()
