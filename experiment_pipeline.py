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
        self.n_epochs = 100
        self.batch_size = 256
        self.learning_rate = 1e-3
       
        self.aug_data_period = 10
        
        # regularization
        self.keep_prob = 0.5
        self.l2 = 0.01
        
        # conv layers: kernel size, n_kernels, pooling, dropout, go to fc, activation_fn
        self.conv_layers = [[1, 3, False, 1.0, False, tf.nn.relu],
                            [5, 32, False, 1.0, False, tf.nn.relu],
                            [5, 32, True, 0.5, True, tf.nn.relu], 
                            [5, 64, False, 1.0, False, tf.nn.relu],
                            [5, 64, True, 0.5, True, tf.nn.relu]]
        
     
        
        # fully connected layers
        self.fc_layers = [512]
        
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
    _keep_prob = tf.placeholder(dtype=tf.float32)
    
    # network structure and prediction
    #_logits = network.basic_lenet(_X, n_classes, _keep_prob)
    _logits = network.lenet_plus(_X, n_classes, param)
    _preds = tf.argmax(_logits, axis=1)
    
    # loss and optimizer
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, labels=_y)
    _loss = tf.reduce_mean(ce)
    train_op = tf.train.AdamOptimizer(learning_rate=param.learning_rate).minimize(_loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # train
    n_batches = n_data // param.batch_size
    best_valid = 0
    best_valid_test = 0
    best_test = 0
    best_test_valid = 0
    for epoch in range(param.n_epochs):
        # re-autment data periodically
        if epoch > 0 and epoch % param.aug_data_period == 0:
            print('re-augment data')
            X_train, y_train, oh_y_train = gen_new_train()
        
        X_train, oh_y_train = shuffle(X_train, oh_y_train)
           
        # training
        epoch_loss = 0
        for batch in range(n_batches):
            bstart = batch * param.batch_size
            bend = (batch+1) * param.batch_size
            
            X_batch = X_train[bstart : bend]
            y_batch = oh_y_train[bstart : bend]
            
            _, loss = sess.run([train_op, _loss], feed_dict={_X:X_batch, _y:y_batch, _keep_prob:param.keep_prob})
            epoch_loss += loss
        
        epoch_loss /= n_batches
        
        # validation
        preds_valid = sess.run(_preds, {_X:X_valid, _keep_prob:1.0})
        valid_accuracy = utils.classification_accuracy(y_valid, preds_valid)
        
        # test
        preds_test = sess.run(_preds, {_X:X_test, _keep_prob:1.0})
        test_accuracy = utils.classification_accuracy(y_test, preds_test)
        
        if valid_accuracy > best_valid:
            best_valid = valid_accuracy
            best_valid_test = test_accuracy
        
        if test_accuracy > best_test:
            best_test = test_accuracy
            best_test_valid = valid_accuracy
        
        print('epoch: ', epoch, ' loss: ', epoch_loss, ' valid accuracy: ', valid_accuracy, ' test accuracy: ', test_accuracy)
    
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