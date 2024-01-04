# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:13:21 2023

@author: savci
"""


#network_creator_beta
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras import Sequential, Model, regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from data_creators import data_create, example_plotter
from initialize_network import *
import numpy as np
import matplotlib.pyplot as plt
import os
'''Parameters for DANN or Source only Network creation'''
'''TODO add option to safe weights with timestamps -> no overwriting, less clutter and confusion'''
save_weights = False    # Save the trained Networks? 
                        # !! Be careful !! to reduce file 
                        # overload saving overwrites old weights with the same setup
                        # 

# domains       MNIST,nMNIST,nMNIST,cMNIST,lMNIST,rMNIST
# num channels      1,     1,     3,     3,     3,     1.
S       = [0,5,5,4] # Sopurce Domain for the experiments; each entry in [0,...,5] 
DAA     = [1,2,4,5] # Target Domain for the experiment;   each entry in [0,...,5]
snr_s   = 3           # Signal-to-Noise Ratio for MNIST-n
angle   = 45          # Rotation Angle for MNIST-r
DA      = False          # True:  Domain Adaptation with DANN metric 
                    # False: Source only Training or
EPOCH   = 200
# depth iterates from depth = l to depth = u-1
# only values 1 to 10 defined
# depth $n$ creates a network with 2 feature extraction layers + $n$ dense layers
l     = 1               # Lower bound depth
u     = 2               # Upper bound depth
plot_source_samples     = False # it does what you would think it does what it is called
plot_target_samples     = False # it does what you would think it does what it is called
plot_one_of_each_domain = True

'''Assert correct parameters'''
assert(len(S)   ==  len(DAA))
assert(0 <  l)
assert(u <= 11)
assert(l <  u)


'''Plot an example of each of the 6 domains'''
if plot_one_of_each_domain:
    example_plotter(1,snr = snr_s, angle = angle)
'''
MAIN: Initialize and train DANN in the loop

'''
model_optimizer = tf.optimizers.legacy.SGD()
print(0)
for depth in range(l,u):
    print(depth)
for i in range(len(S)):
    print(i)
for depth in range(l,u):#s in [3]:
    print(depth)
    for i in range(len(S)):
        print(i)
        target_domain  = DAA[i]
        source_domain   = S[i]
        BATCH_SIZE = 32
        m = 28
        #%% Create and prepare Data
        mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y         = data_create(source_domain, source_domain, snr = snr_s, plotsamples = plot_source_samples)
        mnist_m_train_x, mnist_m_train_y, mnist_m_test_x, mnist_m_test_y = data_create(target_domain, target_domain, snr = snr_s, plotsamples = plot_target_samples)
            #1D,1D,3D,3D,3D,1D
        Cha = [1,1,3,3,3,1]
        '''Make all data 3-channel images'''
        if DA == True:
            if Cha[source_domain]   <   Cha[target_domain]:
                mnist_train_x     = np.repeat(mnist_train_x, 3, axis=3)
                mnist_test_x      = np.repeat(mnist_test_x, 3, axis=3)
            elif Cha[source_domain] >   Cha[target_domain]:
                mnist_m_train_x   = np.repeat(mnist_m_train_x, 3, axis=3)
                mnist_m_test_x    = np.repeat(mnist_m_test_x, 3, axis=3)
        else:
            if Cha[source_domain]   <  3:
                mnist_train_x     = np.repeat(mnist_train_x, 3, axis=3)
                mnist_test_x      = np.repeat(mnist_test_x, 3, axis=3)
            if  Cha[target_domain] < 3:
                mnist_m_train_x   = np.repeat(mnist_m_train_x, 3, axis=3)
                mnist_m_test_x    = np.repeat(mnist_m_test_x, 3, axis=3)
        '''Make data tensorflowable'''
        inputsize = mnist_test_x[1]*mnist_test_x[2]*mnist_test_x[3]
        mnist_train_x     = mnist_train_x.astype('float32')
        mnist_test_x      = mnist_test_x.astype('float32')
        mnist_m_train_x   = mnist_m_train_x.astype('float32')
        mnist_m_test_x    = mnist_m_test_x.astype('float32')
        mnist_train_y_dec = mnist_train_y
        mnist_test_y_dec  = mnist_test_y
        mnist_train_y     = to_categorical(mnist_train_y)
        mnist_test_y      = to_categorical(mnist_test_y)
        mnist_m_train_y_dec = mnist_m_train_y
        mnist_m_test_y_dec  = mnist_m_test_y
        mnist_m_train_y   = to_categorical(mnist_m_train_y)
        mnist_m_test_y    = to_categorical(mnist_m_test_y)
        '''Create Batches'''
        source_dataset   = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y)).shuffle(1000).batch(BATCH_SIZE*2)
        da_dataset       = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y, mnist_m_train_x, mnist_m_train_y)).shuffle(1000).batch(BATCH_SIZE)
        test_dataset     = tf.data.Dataset.from_tensor_slices((mnist_m_test_x, mnist_m_test_y)).shuffle(1000).batch(BATCH_SIZE*2) #Test Dataset over Target Domain
        test_dataset2    = tf.data.Dataset.from_tensor_slices((mnist_m_train_x, mnist_m_train_y)).shuffle(1000).batch(BATCH_SIZE*2) #Test Dataset over Target (used for training)
        domain_labels    = np.vstack([np.tile([1., 0.], [BATCH_SIZE, 1]),np.tile([0., 1.], [BATCH_SIZE, 1])])
        domain_labels    = domain_labels.astype('float32')
        '''Training'''
        model = DANN(depth = depth+1)
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        source_acc     = []  # Source Domain Accuracy while Source-only Training
        da_acc         = []  # Source Domain Accuracy while DA-training
        test_acc       = []  # Testing Dataset (Target Domain) Accuracy 
        test2_acc      = []  # Target Domain (used for Training) Accuracy
        @tf.function
        def train_step_source(s_images, s_labels, lamda=1.0):
                    images = s_images
                    labels = s_labels
                    with tf.GradientTape() as tape:
                        output = model(images, train=True, source_train=True, lamda=lamda)
                        model_loss = get_loss(output, labels)
                        epoch_accuracy(output, labels)
                    gradients_mdan = tape.gradient(model_loss, model.trainable_variables)
                    model_optimizer.apply_gradients(zip(gradients_mdan, model.trainable_variables))

        @tf.function
        def train_step_da(s_images, s_labels, t_images=None, t_labels=None, lamda=1.0):
                    images = tf.concat([s_images, t_images], 0)
                    labels = s_labels
                    with tf.GradientTape() as tape:
                        output = model(images, train=True, source_train=False, lamda=lamda)
                        #print(output)
                        model_loss = get_loss(output[0], labels, output[1], domain_labels)
                        epoch_accuracy(output[0], labels)
                    gradients_mdan = tape.gradient(model_loss, model.trainable_variables)
                    model_optimizer.apply_gradients(zip(gradients_mdan, model.trainable_variables))
                
                
        @tf.function
        def test_step(t_images, t_labels):
                    images = t_images
                    labels = t_labels
                    output = model(images, train=False, source_train=True)
                    epoch_accuracy(output, labels)
                
                
        def train(train_mode, epochs=EPOCH):
                    if train_mode == 'source':
                        dataset = source_dataset
                        train_func = train_step_source
                        acc_list = source_acc
                    elif train_mode == 'domain-adaptation':
                        dataset = da_dataset
                        train_func = train_step_da
                        acc_list = da_acc
                    else:
                        raise ValueError("Unknown training Mode")
                    
                    for epoch in range(epochs):
                        p = float(epoch) / epochs
                        lamda = 2 / (1 + np.exp(-100 * p, dtype=np.float32)) - 1
                        lamda = lamda.astype('float32')
                
                        for batch in dataset:
                            train_func(*batch, lamda=lamda)
                        
                        print("Training: Epoch {} :\t Source Accuracy : {:.3%}".format(epoch, epoch_accuracy.result()), end='  |  ')
                        acc_list.append(epoch_accuracy.result())
                        test()
                        epoch_accuracy.reset_states()
                
                
        def test():
                    epoch_accuracy.reset_states()
                    
                    #Testing Dataset (Target Domain)
                    for batch in test_dataset:
                        test_step(*batch)
                        
                    print("Testing Accuracy : {:.3%}".format(epoch_accuracy.result()), end='  |  ')
                    test_acc.append(epoch_accuracy.result())
                    epoch_accuracy.reset_states()
                    
                    #Target Domain (used for Training)
                    for batch in test_dataset2:
                        test_step(*batch)
                    
                    print("Target Domain Accuracy : {:.3%}".format(epoch_accuracy.result()))
                    test2_acc.append(epoch_accuracy.result())
                    epoch_accuracy.reset_states()
        if DA:
            train('domain-adaptation', EPOCH)
        else:
            train('source', EPOCH)
        x_axis = [i for i in range(0, EPOCH)]
        plt.title('Depth '+str(depth))
        plt.plot(x_axis, source_acc, label="Train Err")
        plt.plot(x_axis, test_acc, label="Domain Test Err")
        plt.plot(x_axis, test2_acc, label="Domain Train Err")
        plt.legend()
        if save_weights:
            if DA:
                path = "trained_networks_SourceOnly"
                isExist = os.path.exists(path)
                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(path)
                model.save_weights('trained_networks_SourceOnly\DANN_S' + str(source_domain) +'_T' + str(target_domain) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'da.h5')
            else:
                path = "trained_networks_DomAdapt"
                isExist = os.path.exists(path)
                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(path)
                model.save_weights('trained_networks_DomAdapt\DANN_S' + str(source_domain) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'source.h5')
        print("iteration done")
        print("depth = {}, source = {}, target = {}".format(depth, source_domain, target_domain))
