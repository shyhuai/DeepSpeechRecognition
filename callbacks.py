import keras
from settings import logger
import numpy as np

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def __init__(self, num_batchs_per_epoch, model):
        super().__init__()
        self.num_batchs_per_epoch = num_batchs_per_epoch
        self.display = 40
        self.avg_loss = [] 
        self.model = model
        self.val_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.avg_loss.append(logs['loss'])
        if batch > 0 and batch % self.display == 0:
            logger.info('For batch {}/{}, loss is {:7.2f}.'.format(batch, self.num_batchs_per_epoch, np.mean(self.avg_loss)))
            self.avg_loss.clear()

    def on_test_batch_end(self, batch, logs=None):
        #logger.info('For batch {}, validation loss is {:7.2f}.'.format(batch, logs['loss']))
        self.val_losses.append(logs['loss'])

    def on_test_end(self, logs=None):
        logger.info('The validation average loss is {:7.2f}.'.format(np.mean(self.val_losses)))
        self.val_losses.clear()

    def on_epoch_end(self, epoch, logs=None):
        logger.info('The average loss for epoch {} is {:7.2f}.'.format(epoch, logs['loss']))


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 80
    if epoch % decay_step == 0 and epoch > 0:
        return lr * decay_rate
    return lr
