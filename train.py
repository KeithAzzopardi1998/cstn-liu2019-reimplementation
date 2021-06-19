from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import cPickle as pickle
import pandas as pd
import math
from datetime import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,LearningRateScheduler,Callback

import models
from keras.optimizers import Adam, SGD
from keras.utils import plot_model as plot
from keras.utils import multi_gpu_model

from metrics import Metrics

from keras import backend as K

from dataset import load_data

# uncomment followng to set fix random seed
np.random.seed(1337)

import argparse

import CSTN as DEMODEL


def get_tensorboard(path):
    tensorboard = TensorBoard(log_dir=path)
    return tensorboard

def save_file(file, path):
    rtcode = os.system(" ".join(["cp", file.replace(".pyc", ".py"), path]))
    assert rtcode == 0

def get_decay(base_lr):
    def step_decay_lr(epoch):
        if epoch < 200:
            return base_lr
        else:
            return base_lr * 0.1 

    return step_decay_lr


def show_score(odmax, score, stage):
    print(stage + ' score: %.6f rmse (real): %.6f mape: %.6f' %
          (score[0], score[1], score[2]))

    print('origin rmse (real): %.6f mape: %.6f' %
          (score[3], score[4]))

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = float(K.get_value(optimizer.lr))
        print('LR: {:.6f}'.format(lr))

def train(lr, batch_size, seq_len, DEMODEL, dataset_path,num_days_test,path_model):

    use_tensorboard = True
    nb_epoch = 200       # number of epoch at training stage
    nb_epoch_cont = 500  # number of epoch at continued training stage
    m_patience = 20

    print("--> TRAINING: loading data...")
    x_train, y_train, x_test, y_test, odmax, map_width, map_height = load_data(data_file=dataset_path,
                                                                timestep=seq_len,
                                                                num_days_test=num_days_test)
    
    """********************************************************************************************"""
    """ Frist, we train our model with fixed learning rate                                         """
    """********************************************************************************************"""
    print("--> TRAINING: initializing training with fixed learning rate")
    
    model_para = {
        "timestep": seq_len,
        "map_height": map_height,
        "map_width": map_width,
        "weather_dim": 0,
        "meta_dim": 0,
    }
    print("---------------- MODEL PARAMETERS ----------------")
    print(model_para)
    print("--------------------------------------------------")

    model = DEMODEL.build_model(**model_para)
    plot(model, to_file=os.path.join(path_model,'networks.png'), show_shapes=True)
    model.summary()
    train_model = model

    # defining the model parameters and compiling it
    loss = DEMODEL.get_loss()
    optimizer = Adam(lr=lr)
    m = Metrics(odmax=odmax)
    metrics = [m.rmse, m.mape, m.o_rmse, m.o_mape]
    train_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    #define the callbacks
    callbacks = []
    hyperparams_name = 'timestep{}.lr{}'.format(seq_len, lr)
    fname_param = os.path.join(path_model, hyperparams_name + '.best.h5')
    lr_logger = SGDLearningRateTracker() # log out the learning rate after a epoch trained
    callbacks.append(lr_logger)
    callbacks.append(EarlyStopping(monitor='val_rmse', patience=m_patience, mode='min'))
    callbacks.append(ModelCheckpoint(
        fname_param, monitor='val_mape', verbose=0, save_best_only=True, mode='min'))
    if use_tensorboard:
        callbacks.append(get_tensorboard(path_model+"/tensorboard-1/"))

    print("--> TRAINING: starting training with fixed learning rate")
    history = train_model.fit(x_train, y_train,
                    nb_epoch=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks,
                    verbose=1)

    print("--> TRAINING: finished training ... saving")
    model.save_weights(os.path.join(
        path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
    train_model.load_weights(fname_param)
    model.save_weights(fname_param, overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_model, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

    model.load_weights(fname_param)           

    """********************************************************************************************"""
    """ Second, we train our model with step_decay learning rate                                   """
    """********************************************************************************************"""
    print("--> TRAINING: initializing training with decaying learning rate")

    # clear session to rebuild the model, in order to switch optimizor
    K.clear_session()
    DEMODEL.clear_graph()

    model = DEMODEL.build_model(**model_para)
    train_model = model

    loss = DEMODEL.get_loss()
    optimizer = Adam(lr=lr)
    m = Metrics(odmax=odmax)
    metrics = [m.rmse, m.mape, m.o_rmse, m.o_mape]
    train_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.load_weights(fname_param)    

    fname_param_step =  os.path.join(
        path_model, \
        hyperparams_name + '.cont.best.h5.{epoch:03d}-{val_mape:.4f}-{val_rmse:.4f}-{val_o_mape:.4f}-{val_o_rmse:.4f}')

    #setting the callbacks for this new training run
    callbacks_cont = []
    callbacks_cont.append(LearningRateScheduler(get_decay(lr)))
    callbacks_cont.append(ModelCheckpoint(
        fname_param_step, monitor='val_mape', verbose=0, save_best_only=False, period=1, save_weights_only=True, mode='min'))
    if use_tensorboard:
        callbacks_cont.append(get_tensorboard(path_model+"/tensorboard-2/"))

    print("--> TRAINING: starting training with decaying learning rate")
    history = train_model.fit(x_train, y_train,
                        nb_epoch=nb_epoch_cont, 
                        batch_size=batch_size,
                        callbacks=callbacks_cont, 
                        validation_data=(x_test, y_test),
                        verbose=1)

    print("--> TRAINING: finished training ... saving")
    pickle.dump((history.history), open(os.path.join(
        path_model, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(
        path_model, '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    model.load_weights(fname_param)
    model.save_weights(fname_param, overwrite=True)
    print("--> TRAINING: done")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='learing rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seq_len', type=int, default=5, help='length of input sequence')
    parser.add_argument('--num_days_test', type=int, default=60, help='number of days in the test set')
    parser.add_argument('--dataset_path', type=str, help='path to the dataset')
    parser.add_argument('--out_path', type=str, help='the directory with the trained model')
    args = parser.parse_args()
    
    if os.path.isdir(args.out_path):
        print("ERROR: the output path already exists")
        print("STOPPING JUST IN CASE")
        exit(1)

    os.makedirs(args.out_path)
    train(lr=args.ls,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            DEMODEL=DEMODEL,
            dataset_path=args.dataset_path,
            num_days_test=args.num_days_test,
            path_model=args.out_path)

