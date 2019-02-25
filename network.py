#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:05:53 2019

@author: mingrui
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, l2_regularizer, max_pool2d, fully_connected, dropout

def lenet_plus(x, n_classes, param):
    # conv layers
    conv_output = []
    for cl in param.conv_layers:
        x = conv2d(x, num_outputs=cl[1], kernel_size=cl[0], stride=1, 
                   weights_regularizer=l2_regularizer(param.l2),
                   activation_fn=tf.nn.relu, padding='VALID')
        
        # pooling
        if cl[2]:
            x = max_pool2d(x, kernel_size=2, stride=2, padding='VALID')
            
        # dropout
        if cl[3] > 0 and cl[3] < 1.0:
            x = dropout(x, keep_prob=cl[3])
            
        # go to fully connected layers
        if cl[4]:
            conv_output.append(flatten(x))
            
    # flatten
    x = tf.concat(conv_output, axis=1)
    
    # fully connected layers
    for fl in param.fc_layers:
        x = fully_connected(x, fl, 
                            weights_regularizer=l2_regularizer(param.l2), 
                            activation_fn=tf.nn.relu)
        x = dropout(x, keep_prob=param.keep_prob)
    
    # output layer
    x = fully_connected(x, num_outputs=n_classes, 
                        weights_regularizer=l2_regularizer(param.l2),
                        activation_fn=None)
    
    return x

# basic lenet with regularization
def reg_lenet(x, n_classes, param):
    # conv layers
    for cl in param.conv_layers:
        x = conv2d(x, num_outputs=cl[1], kernel_size=cl[0], stride=1, 
                   weights_regularizer=l2_regularizer(param.l2),
                   activation_fn=tf.nn.relu, padding='VALID')
    
        x = max_pool2d(x, kernel_size=2, stride=2, padding='VALID')
    
    # fully connected layers
    x = flatten(x)
    for fl in param.fc_layers:
        x = fully_connected(x, fl, 
                            weights_regularizer=l2_regularizer(param.l2), 
                            activation_fn=tf.nn.relu)
        x = dropout(x, keep_prob=param.keep_prob)
    
    # output layer
    x = fully_connected(x, num_outputs=n_classes, 
                        weights_regularizer=l2_regularizer(param.l2),
                        activation_fn=None)
    
    return x

def basic_lenet(x, n_classes, keep_prob):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    n_channels = int(x.get_shape()[3])
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, n_channels, 6), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6))
    x = tf.nn.bias_add(x, b1)

    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16))
    x = tf.nn.bias_add(x, b2)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    x = flatten(x)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    W3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    b3 = tf.Variable(tf.zeros(120))    
    x = tf.add(tf.matmul(x, W3), b3)
    
    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    W4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    b4 = tf.Variable(tf.zeros(84)) 
    x = tf.add(tf.matmul(x, W4), b4)
    
    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = n_classes.
    W5 = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros(n_classes)) 
    logits = tf.add(tf.matmul(x, W5), b5)
    
    return logits

def LeNetPlus(x, n_classes, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.\
    conv1_W = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev = sigma), name="conv1_W")
    conv1_B = tf.Variable(tf.zeros(6), name="conv1_B")
    layer_1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_B

    # TODO: Activation.
    layer_1 = tf.nn.relu(layer_1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma), name="conv2_W")
    conv2_B = tf.Variable(tf.zeros(16), name="conv2_B")
    layer_2 = tf.nn.conv2d(layer_1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_B
    
    # TODO: Activation.
    layer_2 = tf.nn.relu(layer_2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    layer_2 = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Layer 3 : Convolution. output shape = 1x1x400
    #conv3_W = tf.Variable(tf.truncated_normal([5, 5, 16, 400], mean=mu, stddev=sigma), name="conv3_W")
    #conv3_B = tf.Variable(tf.zeros(400), name="conv3_B")
    #layer_3 = tf.nn.conv2d(layer_2, conv3_W, strides=[1,1,1,1], padding='VALID') + conv3_B
    
    
    #layer_3 = tf.nn.relu(layer_3)
    
    # Flatten. input = 5x5x16, output = 400
    layer_1_flat = flatten(layer_1)
    
    # Flatten. Input shape = 1x1x400, output shape = 400
    layer_2_flat = flatten(layer_2)
    
    # Concatentate 2 flattened layers(layer_2_flat and layer_3_flat)
    # Input shape = 1176 + 400, output = 1576
    fc0 = tf.concat([layer_2_flat, layer_1_flat], 1)
    #fc0   = tf.nn.dropout(fc0, keep_prob)
    
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1576, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits