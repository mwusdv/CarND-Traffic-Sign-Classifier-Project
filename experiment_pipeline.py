#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:27:28 2019

@author: mingrui
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

import utils
import network

def data_pipeline():
    X_train, y_train, X_valid, y_valid, X_test, y_test = utils.load_data()
    n_train = len(X_train)
    n_test = len(X_test)
    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    
    image_shape = X_train.shape[1:]
    print("Image data shape =", image_shape)
    
    n_classes = np.max(y_train)+1
    print("Number of classes =", n_classes)

    X_train, y_train, X_valid, y_valid, X_test, y_test = utils.data_process(X_train, y_train, X_valid, y_valid, X_test, y_test)
    n_train = len(X_train)    
    print("Number of augmented training examples =", n_train)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test,
                   n_epochs, batch_size, learning_rate, keep_prob):
    # data
    n_data, n_rows, n_cols, n_channels = X_train.shape
    n_classes = np.max(y_train) + 1
    oh_y_train = utils.one_hot_encode(y_train)
    
    # place holder
    _X = tf.placeholder(dtype=tf.float32, shape=[None, n_rows, n_cols, n_channels])
    _y = tf.placeholder(dtype=tf.uint8, shape=[None, n_classes])    
    _keep_prob = tf.placeholder(dtype=tf.float32)
    
    # network structure and prediction
    _logits = network.basic_lenet(_X, n_classes, _keep_prob)
    _preds = tf.argmax(_logits, axis=1)
    
    # loss and optimizer
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, labels=_y)
    _loss = tf.reduce_mean(ce)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(_loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # train
    n_batches = n_data // batch_size
    for epoch in range(n_epochs):
        X_train, oh_y_train = shuffle(X_train, oh_y_train)
           
        # training
        epoch_loss = 0
        for batch in range(n_batches):
            bstart = batch*batch_size
            bend = (batch+1)*batch_size
            
            X_batch = X_train[bstart : bend]
            y_batch = oh_y_train[bstart : bend]
            
            _, loss = sess.run([train_op, _loss], feed_dict={_X:X_batch, _y:y_batch, _keep_prob:keep_prob})
            epoch_loss += loss
        
        epoch_loss /= n_batches
        
        # validation
        preds_valid = sess.run(_preds, {_X:X_valid, _keep_prob:1.0})
        valid_accuracy = utils.classification_accuracy(y_valid, preds_valid)
        
        # test
        preds_test = sess.run(_preds, {_X:X_test, _keep_prob:1.0})
        test_accuracy = utils.classification_accuracy(y_test, preds_test)
        
        print('epoch: ', epoch, ' loss: ', epoch_loss, ' valid accuracy: ', valid_accuracy, ' test accuracy: ', test_accuracy)
    
    sess.close()

            
    
def experiment():
    # data process
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_pipeline()
    
    # training
    train_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                   n_epochs=100, batch_size=128, learning_rate=1e-3, keep_prob=np.float32(0.5))


if __name__ == '__main__':
    experiment()