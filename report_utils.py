#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 00:03:21 2019

@author: mingrui
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from experiment_pipeline import ExperimentParam
import utils

def hist_eq(img):
    # histogram equalization
    n_rows, n_cols, n_channels = img.shape
    o = np.copy(img)
    for c in range(n_channels):
        o[:, :, c] = cv2.equalizeHist(img[:, :, c])
    
    return o

def show_hist_eq():
    X, y = utils.load_data('./data/train.p')
    
    img1 = X[1000]
    o1 = np.copy(img1)
    for c in range(3):
        o1[:, :, c] = cv2.equalizeHist(img1[:, :, c])
        
    img2 = X[1100]
    o2 = np.copy(img2)
    for c in range(3):
        o2[:, :, c] = cv2.equalizeHist(img2[:, :, c])
    
    
    imgs = [img1, o1, img2, o2]
    titles = ['original', 'pre-processed', 'original', 'pre-processed']
    n_rows = 2
    n_cols = 2
    img_size = 2
    
    hsize = img_size * n_rows
    vsize = img_size * n_cols
    fig = plt.figure(figsize=(hsize,vsize))
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    for i in range(4):
        sub = fig.add_subplot(n_rows, n_cols, i+1)
        sub.set_aspect('auto')
        sub.set_title(titles[i])
        sub.imshow(imgs[i])
        
        sub.axis('off')
        
    plt.show()
    
def show_augment(n_classes, n_each):
    X_train, y_train = utils.load_data('./data/train.p')
    param = ExperimentParam(n_rows=32, n_cols=32, n_channels=3, n_classes=43)
    param._affine_aug_ratio = 1.0
    X_aug, y_aug = utils.augment_data_affine(X_train, y_train, param)
    
    X_train = np.array([hist_eq(X_train[i]) for i in range(len(X_train))])
    X_aug = np.array([hist_eq(X_aug[i]) for i in range(len(X_aug))])
   
    class_indices = np.random.choice(range(43), n_classes, replace=False)
    img_size = 3
    hsize = img_size * n_each
    vsize = img_size * n_classes * 2
    fig = plt.figure(figsize=(hsize,vsize))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    count = 0
    for c in class_indices:
        for i in range(2):
            if i == 0:
                X = X_train
                y = y_train
            else:
                X = X_aug
                y = y_aug
                
            img_indices = np.where(y==c)[0]
            indices = np.random.choice(img_indices, n_each, replace=False)
            for idx in indices:
                count += 1
                sub = fig.add_subplot(n_classes*2, n_each, count)
                sub.set_aspect('auto')
            
                sub.imshow(X[idx])
                sub.text(2,2,str(y[idx]), color='k',backgroundcolor='w')
            
                sub.axis('off')
                
            
    plt.show()

if __name__ == '__main__':
    show_augment(3, 5)