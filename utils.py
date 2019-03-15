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
    #o = o/255 - 0.5
    m = np.mean(o)
    std = np.std(o)+1e-8
    o = (o-m)/std
    
    return o


# rotation, scale, translation, shear
def transorm_image(img, angle_degree, scale, tx, ty, src=None, dst=None):
    n_rows, n_cols = img.shape[:2]
    
    # rotation and scale
    rot_m = cv2.getRotationMatrix2D((n_cols/2, n_rows/2), angle_degree, scale)
    o = cv2.warpAffine(img, rot_m, (n_cols, n_rows))
    
    # translation
    trans_m = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    o = cv2.warpAffine(o, trans_m, (n_cols, n_rows))
    
    # shear
    if src is not None:
        shear_m = cv2.getAffineTransform(src, dst)
        o = cv2.warpAffine(o, shear_m, (n_cols, n_rows))
    
    return o


def gamma_correction(img, gamma=1.0):
    table = np.array([((i/255.0)**gamma)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)

# number of samples in each class
def get_class_size(y):
    n_classes = np.max(y) + 1
    class_size = np.zeros(n_classes, dtype=np.int32)
    for c in range(n_classes):
        class_size[c] = len(np.where(y == c)[0])
        
    return class_size

def distort_test_images(X, param):
    X_distort = []
    n_images, n_rows, n_cols, n_channels = X.shape
    for n in range(n_images):
        # affine transformation
        angle_degree = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.9, 1.1)
        tx, ty = np.random.uniform(-5, 5, 2)
        src = np.array([[5, 5], [5, n_cols-5], [n_rows-5, 5]], dtype=np.float32)
        dst = np.copy(src)
        for p in range(len(dst)):
            delta = np.random.randint(-1, 1, 2)
            dst[p] += delta
        
        I = transorm_image(X[n], angle_degree, scale, tx, ty , src, dst)
        X_distort.append(I)
    
    X_distort = np.array(X_distort)
    
    return X_distort
    
# data augmentation by affine transformation
def augment_data_affine(X, y, param):
    class_size = get_class_size(y)
    n_classes = len(class_size)
    
    N = np.int32(np.max(class_size) * param._affine_aug_ratio)
    
    # affine transfromation
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
                # affine transformation
                angle_degree = np.random.uniform(-10, 10)
                scale = np.random.uniform(0.9, 1.1)
                tx, ty = np.random.uniform(-5, 5, 2)
                src = np.array([[5, 5], [5, 15], [15, 5]], dtype=np.float32)
                dst = np.copy(src)
                for p in range(len(dst)):
                    delta = np.random.randint(-1, 1, 2)
                    dst[p] += delta
                
                I = transorm_image(X[seed_indices[n]], angle_degree, scale, tx, ty , src, dst)
                X_aug.append(I)
                y_aug.append(c)
                
                
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    
    #X = np.concatenate([X, X_aug], axis=0)
    #y = np.concatenate([y, y_aug], axis=0)
    
    return X_aug, y_aug

# data augmentation by gamma correction
def augment_data_gamma(X, y, param):
    class_size = get_class_size(y)
    n_classes = len(class_size)
    
    X_aug = []
    y_aug = []
    for c in range(n_classes):
        # samples in the current class
        class_indices = np.where(y == c)[0]
        
        # randomly sample images for gamma correction
        seed_indices = np.random.choice(class_indices, param._num_gamma_aug, replace=False)
        
        # transform selected images
        for n in range(len(seed_indices)):
            # original image
            img = X[seed_indices[n]]
            
            # gamma correction
            for gamma in param._gammas:
                I = gamma_correction(img, gamma)  
                X_aug.append(I)
                y_aug.append(c)
            
                
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    
    X = np.concatenate([X, X_aug], axis=0)
    y = np.concatenate([y, y_aug], axis=0)
    
    return X, y

# augment data
def augment_data(X, y, param):
    # affine transform
    X_aug, y_aug = augment_data_affine(X, y, param)
    X = np.concatenate([X, X_aug], axis=0)
    y = np.concatenate([y, y_aug], axis=0)
    
    # gamma correction
    #X, y = augment_data_gamma(X, y, param)
    
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
    
def show_classes(n_each):
    X, y = load_data('./data/train.p')
    n_classes = np.max(y)+1
    count = 0
    
    img_size = 3
    hsize = img_size * n_each
    vsize = img_size * n_classes
    fig = plt.figure(figsize=(hsize,vsize))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for c in range(n_classes):
        class_indices = np.where(y==c)[0]
        indices = np.random.choice(class_indices, n_each, replace=False)
        for idx in indices:
            count += 1
            sub = fig.add_subplot(n_classes, n_each, count)
            sub.set_aspect('auto')
        
            sub.imshow(X[idx])
            sub.text(2,2,str(y[idx]), color='k',backgroundcolor='w')
        
            sub.axis('off')
            
    plt.show()
    
# randomly select num_images from X[indices]
# and show them in n_cols columns
def show_images(X, y, indices, n_cols, num_images, preds=None):
    num_images = min(num_images, len(indices))
    n_rows = max(1, num_images // n_cols)
    num_images = min(num_images, n_rows*n_cols)
    
    img_size = 1
    hsize = img_size * n_cols
    vsize = img_size * n_rows
    fig = plt.figure(figsize=(hsize,vsize))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i in range(num_images):
        sub = fig.add_subplot(n_rows, n_cols, i+1)
        sub.set_aspect('auto')
        
        idx = indices[i]
        
        sub.imshow(X[idx])
        sub.text(2,2,str(y[idx]), color='k',backgroundcolor='w')
        
        if preds is not None:
            sub.text(25, 2, str(preds[idx]), color='k',backgroundcolor='w')
            
        sub.axis('off')
        
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

def train_bar_chart():
    X, y = load_data('./data/train.p')
    c = get_class_size(y)
    n_classes = len(c)
    plt.bar(range(n_classes), c, align='center', alpha=0.5, color='b')
    plt.ylabel('class size')
    plt.title('Number of training examples in each class')
    plt.show()
    
if __name__ == '__main__':
    X, y = load_data('./data/train.p')
    X_normed = np.array([pre_process(X[i]) for i in range(len(X))], dtype=np.float32)
    
    indices = np.random.choice(range(X.shape[0]), 5)
    gammas = [0.05, 1.0, 5.0]
    n_gammas = len(gammas)
    img_size = 2
    hsize = img_size * n_gammas
    vsize = img_size * 2
    for idx in indices:
        fig = plt.figure(figsize=(hsize,vsize))
        sub = fig.add_subplot(2, n_gammas+1, 1)
        sub.imshow(X[idx])
        
        sub = fig.add_subplot(2, n_gammas+1, n_gammas+2)
        sub.imshow(X_normed[idx])
        
        for n in range(n_gammas):
            I = gamma_correction(X[idx], gamma=gammas[n])
            
            sub = fig.add_subplot(2, n_gammas+1, n+2)
            sub.imshow(I)
        
            I = pre_process(I)
            sub = fig.add_subplot(2, n_gammas+1, n+3+n_gammas)
            sub.imshow(I)

    plt.show()
   