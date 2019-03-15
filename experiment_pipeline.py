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
import glob
import os
import utils
import network
import PIL

class ExperimentParam:
    def __init__(self, n_rows, n_cols, n_channels, n_classes):
        self._n_epochs = 50
        self._batch_size = 512
        self._learning_rate = 1e-3
        self._momentum = 0.9
       
        self._aug_data_period = 5
        
        # data augmentation
        self._affine_aug_ratio = 2.5
        self._num_gamma_aug = 200
        self._gammas = [0.1, 5.0]
        
        self._valid_period = 1
        
        self._n_test_distortions = 3
        
        # pre-processing layers
        self._pre_prop_layers = [{'kernel': [3, 8], 'pooling': True, 'keep_prob': 1.0, 
                                 'go_to_fc': False, 'activation_fn': tf.nn.relu, 'padding': 'SAME', 
                                 'batch_norm': True, 'l2_reg': 0.01},
                                
                                 {'kernel': [1, 8], 'pooling': True, 'keep_prob': 0.8, 
                                ' go_to_fc': False, 'activation_fn': tf.nn.relu, 'padding': 'SAME', 
                                'batch_norm': True, 'l2_reg': 0.01}]

        # conv layers: 
        self._conv_layers = [{'kernel': [[3, 16], [5, 16], [7, 3, 16], [3, 7, 16]], 'pooling': True, 'keep_prob': 0.8, 
                              'go_to_fc': True, 'activation_fn': tf.nn.relu, 'padding': 'SAME', 'batch_norm': True,
                              'l2_reg': 0.01},
        
                            {'kernel': [[3, 32], [5, 32], [7, 3, 32], [3, 7, 32]], 'pooling': True, 'keep_prob': 1.0, 
                              'go_to_fc': True, 'activation_fn': tf.nn.relu, 'padding': 'SAME', 'batch_norm': True,
                              'l2_reg': 0.01},
        
                            {'kernel': [[3, 64], [5, 2, 64], [2, 5, 64]], 'pooling': True, 'keep_prob': 1.0, 
                              'go_to_fc': True, 'activation_fn': tf.nn.relu, 'padding': 'SAME', 'batch_norm': True,
                              'l2_reg': 0.01}]
        
        # fully connected layers
        self._fc_layers = [{'hidden_dim': 512, 'keep_prob': 0.5, 'activation_fn': tf.nn.relu, 'batch_norm': True,
                            'l2_reg': 0.1}]
                           
                           
        
        #{'hidden_dim': 512, 'keep_prob': 0.5, 'activation_fn': tf.nn.relu, 'batch_norm': True,
        #                    'l2_reg': 0.01}]
        
        # model file name
        self._model_fname = './models/traffic_sign_net'
        
        # image size
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._n_channels = n_channels
        self._n_classes = n_classes
        
def data_pipeline(param):
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
    X_train, y_train = utils.augment_data(X_train, y_train, param)  
    print("Number of augmented training examples =", len(X_train))
    print("Number of validation examples =", len(X_valid))
    
    # pre-process
    X_train = np.array([utils.pre_process(X_train[i]) for i in range(len(X_train))], dtype=np.float32)
    X_valid = np.array([utils.pre_process(X_valid[i]) for i in range(len(X_valid))], dtype=np.float32)
    X_test = np.array([utils.pre_process(X_test[i]) for i in range(len(X_test))], dtype=np.float32)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def gen_new_train(param):
    # load data
    X_train, y_train = utils.load_data('./data/train.p')
    
    # data augmentation
    X_train, y_train = utils.augment_data(X_train, y_train, param)
    
    # pre-process
    X_train = np.array([utils.pre_process(X_train[i]) for i in range(len(X_train))], dtype=np.float32)
    
    # one hot
    oh_y_train = utils.one_hot_encode(y_train)
    
    return X_train, y_train, oh_y_train

    
def train_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test, param):
    # data
    n_data, n_rows, n_cols, n_channels = X_train.shape
    oh_y_train = utils.one_hot_encode(y_train)

    # network structure and prediction
    net = network.TrafficSignNet(param)   
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        #train_op = tf.train.AdamOptimizer(learning_rate=param._learning_rate).minimize(net._loss)
        train_op = tf.train.RMSPropOptimizer(learning_rate=param._learning_rate, momentum=param._momentum).minimize(net._loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
   
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
            X_train, y_train, oh_y_train = gen_new_train(param)
        
        indices = shuffle(np.array(range(n_data)))
        
        # training
        epoch_loss = 0
        for batch in range(n_batches):
            bstart = batch * param._batch_size
            bend = (batch+1) * param._batch_size
            
            batch_indices = indices[bstart:bend]
            X_batch = X_train[batch_indices]
            y_batch = oh_y_train[batch_indices]
            
            _, loss = sess.run([train_op, net._loss], 
                               feed_dict={net._X:X_batch, net._y:y_batch, net._is_training:True})
            epoch_loss += loss
        
        epoch_loss /= n_batches
        
        # turn off traning flag to calculate predictions
        if epoch % param._valid_period == 0 or epoch == param._n_epochs-1: 
            # validation
            preds_valid = sess.run(net._preds, {net._X:X_valid, net._is_training:False})
            valid_accuracy = utils.classification_accuracy(y_valid, preds_valid)
            
            # test
            
            preds_test = sess.run(net._preds, {net._X:X_test, net._is_training:False})
            test_accuracy = utils.classification_accuracy(y_test, preds_test)
            
            #preds_test1 = classify(sess, X_test, net, param)
            #test_accuracy1 = utils.classification_accuracy(y_test, preds_test1)
            
            if valid_accuracy > best_valid:
                best_valid = valid_accuracy
                best_valid_test = test_accuracy
                saver.save(sess, param._model_fname)
            
            if test_accuracy > best_test:
                best_test = test_accuracy
                best_test_valid = valid_accuracy
        
            print('epoch: ', epoch, ' loss: ', epoch_loss, ' valid accuracy: ', valid_accuracy, \
              '     test accuracy: ', test_accuracy) #, 'test accuracy1: ', test_accuracy1)
        else:
            print('epoch: ', epoch, ' loss: ', epoch_loss)
        
    
    sess.close()
    print('best valid: ', best_valid, ' best valid test: ', best_valid_test)
    print('best test: ', best_test, ' best_test_valid: ', best_test_valid)


# load model
def load_model(model_fname, param):
    # define model
    tf.reset_default_graph()
    net = network.TrafficSignNet(param)
    
    # load model data
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_fname)
    
    return net, sess

# load and test model
def test_model(model_fname, param, X, y):
    net, sess = load_model(model_fname, param)
    n_images = X.shape[0]
    batch_size = param._batch_size
    n_batches = n_images // batch_size
    if n_batches * batch_size < n_images:
        n_batches += 1
    
    preds = []
    for batch in range(n_batches):
        bt_start = batch * batch_size
        bt_end = min(bt_start + batch_size, n_images)
        X_batch = X[bt_start : bt_end]
        bt_preds = sess.run(net._preds, {net._X:X_batch, net._is_training:False})
        preds.append(bt_preds)

    preds = np.concatenate(preds)
    accuracy = utils.classification_accuracy(y, preds)
    
    sess.close()
    return accuracy, preds
    
    
# show bad cases in the test data
def show_bad_cases(test_fname, param):
    # load test data
    X_test, y_test = utils.load_data(test_fname)
    X_test_normed = np.array([utils.pre_process(X_test[i]) for i in range(len(X_test))], dtype=np.float32)
    
    n_data, n_rows, n_cols, n_channels = X_test.shape
    param._n_rows = n_rows
    param._n_cols = n_cols
    param._n_channels = n_channels
    
    # load model
    n_classes = int(np.max(y_test) + 1)
    tf.reset_default_graph()
    net = network.TrafficSignNet(n_classes, param)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    saver.restore(sess, param._model_fname)
    
    # test
    preds_test = sess.run(net._preds, {net._X:X_test_normed, net._is_training:False})
    test_accuracy = utils.classification_accuracy(y_test, preds_test)
    print('test accuracy: ', test_accuracy)
    sess.close()
    X_test_normed = None
    
    # show test images that are not correctly classified
    err_indices = np.where(preds_test != y_test)[0]
    utils.show_images(X_test, y_test, err_indices, n_cols=5, num_images=200, preds=preds_test)
    
def experiment():
    param = ExperimentParam(n_rows=32, n_cols=32, n_channels=3, n_classes=43)
    
    # data process
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_pipeline(param)
    
    # training
    start = datetime.datetime.now()
    train_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test, param)
    end = datetime.datetime.now()
    delta = (end - start).seconds
    print(delta, ' seoncds.')
    
    # test
    model_fname = param._model_fname
    test_accuracy, _= test_model(model_fname, param, X_test, y_test)
    print('Test accuracy: ', test_accuracy)
    

# load images from web
def load_web_images(folder, param):
    X = []
    y = []
    img_fname_list = glob.glob(os.path.join(folder, '*.jpg'))
    for fname in img_fname_list:
        img = PIL.Image.open(fname)
        img = img.resize([32, 32], PIL.Image.BILINEAR)
        X.append(np.array(img, dtype=np.uint8))

        label = int(fname.split('/')[-1].split('.')[0].split('-')[-1])
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# experiments on the images downloaded from web
def exp_web_images():
    # parameters
    param = ExperimentParam(n_rows=32, n_cols=32, n_channels=3, n_classes=43)
    
    # load data
    folder = './from-web'
    X, y = load_web_images(folder, param)
    X = np.array([utils.pre_process(X[i]) for i in range(len(X))], dtype=np.float32)
 
    # load model
    model_fname = param._model_fname
    net, sess = load_model(model_fname, param)
   
    preds, softmax = sess.run([net._preds, net._softmax], {net._X:X, net._is_training:False})
    accuracy = utils.classification_accuracy(y, preds)
    print('Accuracy on web images: ', accuracy)
    print('labels: ', y)
    print('predictions: ', preds)
    
    
    # top softmax
    topk = sess.run(tf.nn.top_k(tf.constant(softmax), k=3))
    print(topk)
    
 # experiments on the test data
def exp_test_data():
    # parameters
    param = ExperimentParam(n_rows=32, n_cols=32, n_channels=3, n_classes=43)
    
    # load data
    X_test, y_test = utils.load_data('./data/test.p')
    X_test = np.array([utils.pre_process(X_test[i]) for i in range(len(X_test))], dtype=np.float32)
 
    # load model
    model_fname = param._model_fname

    #preds = sess.run(net._preds, {net._X:X_test, net._is_training:False})
    accuracy, _ = test_model(model_fname, param, X_test, y_test)
    print('Test accuracy: ', accuracy)
    
    
if __name__ == '__main__':
    mode = 2
    
    if mode == 0:
        experiment()
    elif mode == 1:
        exp_web_images()
    elif mode == 2:
        exp_test_data()
    elif mode == 3:
        param = ExperimentParam()
        test_fname = './data/test.p'
        show_bad_cases(test_fname, param)
        utils.show_classes(5)