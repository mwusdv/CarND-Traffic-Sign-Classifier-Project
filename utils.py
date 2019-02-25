#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:46:24 2019

@author: mingrui
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# image pre-processing
def pre_process(img):
    # histogram equalization
    n_rows, n_cols, n_channels = img.shape
    o = np.zeros([n_rows, n_cols, n_channels], dtype=np.float32)
    for c in range(n_channels):
        o[:, :, c] = cv2.equalizeHist(img[:, :, c])
        
    # normalization
    o = o/255 - 0.5
    
    return o


# rotation, scale, translation, shear
def transorm_image(img, angle_degree, scale, tx, ty, src, dst):
    n_rows, n_cols = img.shape[:2]
    
    # rotation and scale
    rot_m = cv2.getRotationMatrix2D((n_cols/2, n_rows/2), angle_degree, scale)
    o = cv2.warpAffine(img, rot_m, (n_cols, n_rows))
    
    # translation
    trans_m = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    o = cv2.warpAffine(o, trans_m, (n_cols, n_rows))
    
    # shear
    shear_m = cv2.getAffineTransform(src, dst)
    o = cv2.warpAffine(o, shear_m, (n_cols, n_rows))
    
    return o

# number of samples in each class
def get_class_size(y):
    n_classes = np.max(y) + 1
    class_size = np.zeros(n_classes, dtype=np.int32)
    for c in range(n_classes):
        class_size[c] = len(np.where(y == c)[0])
        
    return class_size
        
# data augmentation
def augment_data(X, y):
    class_size = get_class_size(y)
    n_classes = len(class_size)
    
    # augment data, such that each class has N samples
    N = np.max(class_size)
    N = int(N*1.8)
    
    X_aug = []
    y_aug = []
    for c in range(n_classes):
        num_new_samples = N - class_size[c]
        
        if num_new_samples > 0:
            # samples in the current class
            class_indices = np.where(y == c)[0]
            
            # randomly sample images to transform
            seed_indices = np.random.choice(class_indices, num_new_samples, replace=(num_new_samples > class_size[c]))
            
            # transform selected images
            for n in range(len(seed_indices)):
                
                angle_degree = np.random.uniform(-10, 10)
                scale = np.random.uniform(0.9, 1.1)
                tx, ty = np.random.uniform(-3, 3, 2)
                src = np.array([[5, 5], [5, 15], [15, 5]], dtype=np.float32)
                dst = np.copy(src)
                for p in range(3):
                    delta = np.random.randint(-2, 2, 2)
                    dst[p] += delta
                
                I = transorm_image(X[seed_indices[n]], angle_degree, scale, tx, ty , src, dst)
                X_aug.append(I)
                y_aug.append(c)
                
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    
    X = np.concatenate([X, X_aug], axis=0)
    y = np.concatenate([y, y_aug], axis=0)
    
    return X, y

def plot_samples(n_row,n_col,X,y):
    plt.figure(figsize = (5,5))
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        ax.set_aspect('equal')
        ind_plot = np.random.randint(1,len(y))
        
        plt.imshow(X[ind_plot])
        plt.text(2,4,str(y[ind_plot]),
             color='k',backgroundcolor='w')
        plt.axis('off')
    plt.show()
    

def classification_accuracy(labels, predictions):
    corrects = np.sum(labels == predictions)
    return corrects / len(labels)


def one_hot_encode(y):
    n_classes = np.max(y) + 1
    oh = np.eye(n_classes)[y]
    
    return oh

# load data
def load_data(data_fname):
    fd = open(data_fname, 'rb')
    data = pickle.load(fd)
    fd.close()
    
    X, y = data['features'], data['labels']
    return X, y
