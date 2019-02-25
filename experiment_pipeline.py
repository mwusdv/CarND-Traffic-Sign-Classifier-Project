#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:27:28 2019

@author: mingrui
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import datetime

import utils
import network

class TrainParam:
    def __init__(self):
        self._n_epochs = 100
        self._batch_size = 256
        self._learning_rate = 1e-3
        self._momentum = 0.9
       
        self._aug_data_period = 10
        
        # regularization
        self._l2 = 0.01
        
        # pre-processing layers
        #  kernel size, n_kernels, pooling, dropout, go to fc, activation_fn, padding, batch norm
        self._pre_prop_layers = [[3, 8, False, 1.0, False, tf.nn.relu, 'SAME'],
                                [1, 8, False, 1.0, False, tf.nn.relu, 'SAME']]
        # conv layers: 
        # kernel size, n_kernels, pooling, dropout, go to fc, activation_fn, padding
        self._conv_layers0 = [[5, 32, False, 1.0, False, tf.nn.relu],
                            [5, 32, True, 0.5, True, tf.nn.relu], 
                            [5, 64, False, 1.0, False, tf.nn.relu],
                            [5, 64, True, 0.5, True, tf.nn.relu]]
        
     
        self._conv_layers = [[5, 32, True, 1.0, True, tf.nn.relu, 'SAME', True],
                            [5, 64, True, 0.9, True, tf.nn.relu, 'SAME', True], 
                            [5, 128, True, 0.8, True, tf.nn.relu, 'SAME', True]]
                            
        
        # fully connected layers
        # hidden_dim, dropout, activation_fn, batch_norm
        self._fc_layers = [[1024, 0.5, tf.nn.relu, True]]
        
def data_pipeline():
    # load data
    X_train, y_train = utils.load_data('./data/train.p')
    X_valid, y_valid = utils.load_data('./data/valid.p')
    X_test, y_test = utils.load_data('./data/test.p')
    
    n_train = len(X_train)
    n_test = len(X_test)
    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    
    image_shape = X_train.shape[1:]
    print("Image data shape =", image_shape)
    
    n_classes = np.max(y_train)+1
    print("Number of classes =", n_classes)

    # data augmentation
    X_train, y_train = utils.augment_data(X_train, y_train)
    n_train = len(X_train)    
    print("Number of augmented training examples =", n_train)
    
    # pre-process
    X_train = np.array([utils.pre_process(X_train[i]) for i in range(len(X_train))], dtype=np.float32)
    X_valid = np.array([utils.pre_process(X_valid[i]) for i in range(len(X_valid))], dtype=np.float32)
    X_test = np.array([utils.pre_process(X_test[i]) for i in range(len(X_test))], dtype=np.float32)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def gen_new_train():
    # load data
    X_train, y_train = utils.load_data('./data/train.p')
    
    # data augmentation
    X_train, y_train = utils.augment_data(X_train, y_train)
    
    # pre-process
    X_train = np.array([utils.pre_process(X_train[i]) for i in range(len(X_train))], dtype=np.float32)
    
    # one hot
    oh_y_train = utils.one_hot_encode(y_train)
    
    return X_train, y_train, oh_y_train

    
def train_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test, param):
    # data
    n_data, n_rows, n_cols, n_channels = X_train.shape
    n_classes = int(np.max(y_train) + 1)
    oh_y_train = utils.one_hot_encode(y_train)
    
    # place holder
    _X = tf.placeholder(dtype=tf.float32, shape=[None, n_rows, n_cols, n_channels])
    _y = tf.placeholder(dtype=tf.uint8, shape=[None, n_classes])    

    # network structure and prediction
    net = network.TrafficSignNet(n_classes, param)
    
    net.set_training(True)
    _logits = net.logits(_X)
    _preds = tf.argmax(_logits, axis=1)
    
    # loss and optimizer
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, labels=_y)
    _loss = tf.reduce_mean(ce)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        #train_op = tf.train.AdamOptimizer(learning_rate=param._learning_rate).minimize(_loss)
        train_op = tf.train.RMSPropOptimizer(learning_rate=param._learning_rate, momentum=param._momentum).minimize(_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # train
    n_batches = n_data // param._batch_size
    best_valid = 0
    best_valid_test = 0
    best_test = 0
    best_test_valid = 0
    for epoch in range(param._n_epochs):
        # re-autment data periodically
        if epoch > 0 and epoch % param._aug_data_period == 0:
            print('re-augment data')
            X_train, y_train, oh_y_train = gen_new_train()
        
        X_train, oh_y_train = shuffle(X_train, oh_y_train)
           
        # turn on training flag
        net.set_training(True)
        
        # training
        epoch_loss = 0
        for batch in range(n_batches):
            bstart = batch * param._batch_size
            bend = (batch+1) * param._batch_size
            
            X_batch = X_train[bstart : bend]
            y_batch = oh_y_train[bstart : bend]
            
            _, loss = sess.run([train_op, _loss], feed_dict={_X:X_batch, _y:y_batch})
            epoch_loss += loss
        
        epoch_loss /= n_batches
        
        # turn off traning flag to calculate predictions
        net.set_training(False)
        
        # validation
        preds_valid = sess.run(_preds, {_X:X_valid})
        valid_accuracy = utils.classification_accuracy(y_valid, preds_valid)
        
        # test
        preds_test = sess.run(_preds, {_X:X_test})
        test_accuracy = utils.classification_accuracy(y_test, preds_test)
        
        if valid_accuracy > best_valid:
            best_valid = valid_accuracy
            best_valid_test = test_accuracy
        
        if test_accuracy > best_test:
            best_test = test_accuracy
            best_test_valid = valid_accuracy
        
        print('epoch: ', epoch, ' loss: ', epoch_loss, ' valid accuracy: ', valid_accuracy, \
              ' test accuracy: ', test_accuracy)
    
    sess.close()
    print('best valid: ', best_valid, ' best valid test: ', best_valid_test)
    print('best test: ', best_test, ' best_test_valid: ', best_test_valid)

            
    
def experiment():
    # data process
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_pipeline()
    
    # training
    start = datetime.datetime.now()
    param = TrainParam()
    train_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test, param)
    end = datetime.datetime.now()
    delta = (end - start).seconds
    print(delta, ' seoncds.')

if __name__ == '__main__':
    experiment()