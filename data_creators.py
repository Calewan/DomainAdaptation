# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:07:29 2023

@author: savci
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from PIL import Image
#import os
#os.chdir('/p/sys/department/Masterarbeit_Can/MTHESIS/AlexPaperCode')
#os.chdir('P:\department\Masterarbeit_Can\MTHESIS\AlexPaperCode')
from scipy.ndimage import rotate, shift
'''
Creator functions for the Domains 
     > noisy mnist (1 Channel)
     > noisy mnist (3 Channels)
     > colored mnist (3 Channel)
     > colored & structured mnist (3 Channel)
     > rotated mnist (3 Channel)
Data augmentation functionality
'''

def create_nMNIST_1D(X,snr):
    '''Input MNIST (-1,28,28,1) with pixels in [0,1]'''
    batch_size = X.shape[0]
    X       = X.reshape(batch_size, -1)
    n       = X.shape[1]
    batch   = np.zeros((batch_size, n))
    for i in range(batch_size):
        std = X[i,:].std()
        image = np.random.normal(0,std/snr,(n)).astype('float32')
        image = image + X[i,:]
        batch[i] = image
    batch = batch.reshape((batch_size,28,28,1))
    return batch

def create_nMNIST_3D(X,snr):
    '''Input MNIST (-1,28,28,1) with pixels in [0,1]'''
    batch_size  = X.shape[0]
    X           = np.repeat(X, 3, axis=3)
    X           = X.reshape(batch_size, -1)
    n           = X.shape[1]
    batch       = np.zeros((batch_size, n))
    for i in range(batch_size):
        std         = X[i,:].std()
        image       = np.random.normal(0,std/snr,(n)).astype('float32')
        image       = image + X[i,:]
        batch[i]    = image
    batch       = batch.reshape((batch_size,28,28,3))
    return batch

def create_cMNIST_3D(X):
    '''Input MNIST (-1,28,28,1) with pixels in [0,1]'''
    batch_size  = X.shape[0]
    X           = np.repeat(X, 3, axis=3)
    # Convert the MNIST images to binary
    batch_binary = (X > 0.4)
    # Create a new placeholder variable for our batch
    batch = np.zeros((batch_size,28, 28, 3))
    for i in range(batch_size):
        r = randint(0,255)
        g = randint(0,255)
        b = randint(0,255)
        image = np.ones((28,28,3))
        image[:,:,0] = r
        image[:,:,1] = g
        image[:,:,2] = b
        # Conver the image to float between 0 and 1
        image = np.asarray(image) / 255.0
        for j in range(3):
            image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0    
        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]
        batch[i] = image
        batch = batch.reshape((batch_size,28,28,3))
    return batch


def create_lMNIST_3D(X):
    '''Input MNIST (-1,28,28,1) with pixels in [0,1]'''
    batch_size  = X.shape[0]
    X           = np.repeat(X, 3, axis=3)
    # Convert the MNIST images to binary
    batch_binary = (X > 0.4)
    # Create a new placeholder variable for our batch
    batch = np.zeros((batch_size, 28, 28, 3))
    #lena = Image.open('/p/sys/department/Masterarbeit_Can/MTHESIS/AlexPaperCode/Resources/lena.png')#
    lena = Image.open('Resources\lena.png')
    for i in range(batch_size):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 28)
        y_c = np.random.randint(0, lena.size[1] - 28)
        image = lena.crop((x_c, y_c, x_c + 28, y_c + 28))
        # Conver the image to float between 0 and 1
        image = np.asarray(image) / 255.0
        for j in range(3):
            image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0
        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]
        batch[i] = image
        batch = batch.reshape((batch_size,28,28,3))
    return batch

def create_rotMNIST_1D(X,angle):
    '''Input MNIST (-1,28,28,1) with pixels in [0,1]'''
    batch_size = X.shape[0]
    batch   = np.zeros((batch_size, 28,28,1))
    for i in range(batch_size):
        r_rotated = rotate(X[i], angle, reshape=False)
        batch[i] = r_rotated
    batch = batch.reshape((batch_size,28,28,1))
    return batch

def augment_data(X,y,rotation_range=20, zoom_range=0.1, shift_range=0.1):
    augmented_images = []
    augmented_labels = []
    num_images = X.shape[0]
    CHANNELS = X.shape[3]
    for i in range(num_images):
        image = X[i]
        label = y[i]
        # Randomly rotate each channel
        angle = np.random.uniform(low=-rotation_range, high=rotation_range)
        r_rotated = rotate(image, angle, reshape=False)
        # Randomly shift each channel
        shift_factor = np.random.uniform(low=-shift_range, high=shift_range, size=(2,))
        r_shifted = shift(r_rotated, shift_factor)
        augmented_images.append(r_shifted)
        augmented_labels.append(label)
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    augmented_images = np.stack(augmented_images, axis=0)
    augmented_labels = np.stack(augmented_labels, axis=0)
    x_train_augmented = np.concatenate([X, augmented_images])
    y_train_augmented = np.concatenate([X, augmented_labels])
    x_train_augmented = x_train_augmented.reshape((-1, 28, 28,CHANNELS))
    return augmented_images, augmented_labels

''' 
MAIN FUNCTIONS 
    data_create 
        > create Source and Target data as MNIST variants
    example_plotter
        > plot a random MNIST datapoint and all its domain variants
    '''


def data_create(mode_train, mode_test, angle = 45, snr = 1, plotsamples = True, augment = False):
    '''modes :  
        0 - MNIST
        1 - nMNIST 1D
        2 - nMNIST 3D
        3 - cMNIST 3D
        4 - lMNIST 3D
        5 - rotMNIST 1D
                
       Want to double the Data by augmentation?'''
    (X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr.reshape((60000, 28, 28,1))
    X_te = X_te.reshape((10000, 28, 28,1))
    X_tr, X_te = X_tr / 255.0, X_te / 255.0
    if mode_train == 1:
        X_tr = create_nMNIST_1D(X_tr,snr)
    elif mode_train == 2:
        X_tr = create_nMNIST_3D(X_tr,snr)
    elif mode_train == 3:
        X_tr = create_cMNIST_3D(X_tr)
    elif mode_train == 4:
        X_tr = create_lMNIST_3D(X_tr)
    elif mode_train == 5:
        X_tr = create_rotMNIST_1D(X_tr,angle)
    if mode_test == 1:
        X_te = create_nMNIST_1D(X_te,snr)
    elif mode_test == 2:
        X_te = create_nMNIST_3D(X_te,snr)
    elif mode_test == 3:
        X_te = create_cMNIST_3D(X_te)
    elif mode_test == 4:
        X_te = create_lMNIST_3D(X_te)
    elif mode_test == 5:
        X_te = create_rotMNIST_1D(X_te,angle)
        
    if augment == True:
        X_tr, y_tr = augment_data(X_tr, y_tr)
        
    if plotsamples == True:
        cha     = X_tr.shape[3]
        plt.figure(figsize=(3,4))
        for i in range(6):
                plt.subplot(2,3, i+1)
                rnd = randint(0,60000)
                X = X_tr[rnd,:,:,:]
                if cha == 1:
                    plt.imshow(X,cmap= 'gray')
                else:
                    plt.imshow(X)    
                plt.axis('off')
        plt.show()
        cha     = X_te.shape[3]
        plt.figure(figsize=(3,4))
        for i in range(6):
                plt.subplot(2,3, i+1)
                rnd = randint(0,10000)
                X = X_te[rnd,:,:,:]
                X = X.reshape(28,28,-1)
                if cha == 1:
                    plt.imshow(X,cmap= 'gray')
                else:
                    plt.imshow(X) 
                plt.axis('off')
        plt.show()
    X_tr, X_te = X_tr.astype('float32'), X_te.astype('float32')
    return X_tr, y_tr, X_te, y_te

def example_plotter(num,angle = 45,snr=1):
    (X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr.reshape((60000, 28, 28,1))
    X_te = X_te.reshape((10000, 28, 28,1))
    X_tr, X_te = X_tr / 255.0, X_te / 255.0
    X_tr, X_te = X_tr.astype('float32'), X_te.astype('float32')
    for i in range(num):
        plt.figure()
        rnd = randint(0,60000)
        X = X_tr[rnd,:,:,:]
        cha = 1
        plt.subplot(2,3,1)
        plt.imshow(X,cmap= 'gray')
        plt.axis('off')
        x = create_nMNIST_1D(X.reshape(1,28,28,-1),snr).reshape(28,28,1)
        plt.subplot(2,3,2)
        plt.imshow(x,cmap= 'gray')
        plt.axis('off')
        x = create_nMNIST_3D(X.reshape(1,28,28,-1), snr).reshape(28,28,3)
        plt.subplot(2,3,3)
        plt.imshow(x)
        plt.axis('off')
        x = create_cMNIST_3D(X.reshape(1,28,28,-1)).reshape(28,28,3) 
        plt.subplot(2,3,4)
        plt.imshow(x)
        plt.axis('off')
        x = create_lMNIST_3D(X.reshape(1,28,28,-1)).reshape(28,28,3) 
        plt.subplot(2,3,5)
        plt.imshow(x)
        plt.axis('off')
        x = create_rotMNIST_1D(X.reshape(1,28,28,-1),angle).reshape(28,28,1) 
        plt.subplot(2,3,6)
        plt.imshow(x,cmap= 'gray')
        plt.axis('off')
        plt.show()
    return None