#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:05:53 2019

@author: mingrui
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, l2_regularizer, max_pool2d, fully_connected, dropout, batch_norm

class TrafficSignNet:
    def __init__(self, n_classes, param):
        self._is_training = True
        self._param = param
        self._n_classes = n_classes
        
        # place holder
        self._X = tf.placeholder(dtype=tf.float32, shape=[None, param._n_rows, param._n_cols, param._n_channels])
        self._y = tf.placeholder(dtype=tf.uint8, shape=[None, n_classes])    
      
        # logits and predictions
        self._logits = self.logits()
        self._preds = tf.argmax(self._logits, axis=1)
        
        self._softmax = tf.nn.softmax(logits=self._logits)
        epsilon = 1e-6
        entropy = -tf.reduce_sum(self._softmax * tf.log(self._softmax + epsilon), axis=1)
        self._entropy = tf.nn.relu(entropy)
        
        # loss
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._logits, labels=self._y)
        self._loss = tf.reduce_mean(ce)
        
    def set_training(self, flag):
        self._is_training = flag
        
    def logits(self):
        # preprocess layers
        preprop_output = self.pre_process_layers(self._X)
        
        # convolution layers
        conv_output, conv_2_fc = self.conv_layers(preprop_output)
     
        # fully connected layers
        x = conv_2_fc
        for layer in self._param._fc_layers:
            # batch norm
            if layer['batch_norm']:
                bn = batch_norm
                bn_params = {'center':True, 'scale':True, 'is_training':self._is_training}
            else:
                bn = None
                bn_params = None
                
            x = fully_connected(x, layer['hidden_dim'], 
                                weights_regularizer=l2_regularizer(layer['l2_reg']), 
                                activation_fn=layer['activation_fn'],
                                normalizer_fn=bn, 
                                normalizer_params=bn_params)
            
            
            x = dropout(x, keep_prob=layer['keep_prob'], is_training=self._is_training)
        
        # output layer
        x = fully_connected(x, num_outputs=self._n_classes, 
                            weights_regularizer=l2_regularizer(layer['l2_reg']),
                            activation_fn=None)
        
        return x
    
    # preprocess layers
    def pre_process_layers(self, layer_input):
        # pre-processing layers
        x = layer_input 
        for layer in self._param._pre_prop_layers:
            # batch norm
            if layer['batch_norm']:
                bn = batch_norm
                bn_params = {'center':True, 'scale':True, 'is_training':self._is_training}
            else:
                bn = None
                bn_params = None
                
            kn = layer['kernel']
            kernel_size = kn[0]
            num_outputs = kn[1]
            
            x = conv2d(x, num_outputs=num_outputs, kernel_size=kernel_size, stride=1, 
                       weights_regularizer=l2_regularizer(layer['lr_reg']),
                       activation_fn=layer['activation_fn'], padding=layer['padding'],
                       normalizer_fn=bn, 
                       normalizer_params=bn_params)

        return x
    
    def conv_layers(self, layer_input):
        # conv channels
        conv_2_fc = []
        prev_x = layer_input
        for layer in self._param._conv_layers:
            # batch norm
            if layer['batch_norm']:
                bn = batch_norm
                bn_params = {'center':True, 'scale':True, 'is_training':self._is_training}
            else:
                bn = None
                bn_params = None
                
            # convlution with  each kernel size
            layer_output = []
            for kn in layer['kernel']:
                # get output from the previous layer
                x = prev_x
                
                # conv kernel
                if len(kn) == 2:
                    kernel_size = kn[0]
                    num_outputs = kn[1]
                else:
                    kernel_size = kn[:2]
                    num_outputs = kn[2]
                    
                # convlutions
                x = conv2d(x, num_outputs=num_outputs, kernel_size=kernel_size, stride=1, 
                           weights_regularizer=l2_regularizer(layer['l2_reg']),
                           activation_fn=layer['activation_fn'], padding=layer['padding'],
                           normalizer_fn=bn, 
                           normalizer_params=bn_params)
                
                # pooling
                if layer['pooling']:
                    x = max_pool2d(x, kernel_size=2, stride=2, padding=layer['padding'])
                    
                # dropout
                if layer['keep_prob'] > 0 and layer['keep_prob'] < 1.0:
                    x = dropout(x, keep_prob=layer['keep_prob'], is_training=self._is_training)
                    
                # record output with current kernel size
                layer_output.append(x)
            
            x = tf.concat(layer_output, axis=3)
            
            # go to fully connected layers
            if layer['go_to_fc']:
                conv_2_fc.append(flatten(x))
                
            # go to next layer
            prev_x = x
            
        # flatten
        conv_2_fc = tf.concat(conv_2_fc, axis=1)
        conv_output = prev_x
        
        return conv_output, conv_2_fc
        