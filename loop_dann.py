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
import numpy as np
import matplotlib.pyplot as plt

'''Parameters for DANN or Source only Network creation'''
'''TODO add option to safe weights with timestamps -> no overwriting, less clutter and confusion'''
safe_weights = True    # Safe the trained Networks? 
                        # !! Be careful !! to reduce file 
                        # overload saving overwrites old weights with the same setup
                        # 

# domains       MNIST,nMNIST,nMNIST,cMNIST,lMNIST,rMNIST
# num channels      1,     1,     3,     3,     3,     1.
S       = [0]#[0,5,5,4] # Sopurce Domain for the experiments; each entry in [0,...,5] 
DAA     = [1]#[1,2,4,5] # Target Domain for the experiment;   each entry in [0,...,5]
snr_s = 3           # Signal-to-Noise Ratio for MNIST-n
angle = 45          # Rotation Angle for MNIST-r
DA = False          # True:  Domain Adaptation with DANN metric 
                    # False: Source only Training or
# depth iterates from depth = l to depth = u-1
# only values 1 to 10 defined
# depth $n$ creates a network with 2 feature extraction layers + $n$ dense layers
l = 1               # Lower bound depth
u = 2               # Upper bound depth
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
'''All about DANN'''
#Gradient Reversal Layer
@tf.custom_gradient
def reverse_grad(x, grad_relaxation=1.0):
    y = tf.identity(x)
    
    def grad(dy):
        return grad_relaxation * -dy, None
    
    return y, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x, grad_relaxation=1.0):
        return reverse_grad(x, grad_relaxation)


class DANN(Model):
    '''
    Initialize a DANN Model with feature extraction  and different depth label classification networks
    TODO: Depth n initializes network of depth 3 + regularization. More shallow networks need to be hardcoded right now
    '''
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        #Feature Extractor
        self.feature_extractor_layer0 = Conv2D(32, kernel_size=(3, 3), activation='relu')#Output size (1,26,26,32)
        self.feature_extractor_layer1 = BatchNormalization()
        self.feature_extractor_layer2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))#Output size (1,13,13,32)
        
        self.feature_extractor_layer3 = Conv2D(64, kernel_size=(5, 5), activation='relu')#Output size (2,9,9,64)
        self.feature_extractor_layer4 = Dropout(0.5)
        self.feature_extractor_layer5 = BatchNormalization()
        self.feature_extractor_layer6 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))#Output size (1,4,4,64)
        
        #Label Predictor
        if depth == 1:
            self.label_predictor_layer0 = Dense(10, activation=None)
        elif depth == 2:
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(10, activation=None)
        elif depth == 3:    
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1 = Dropout(0.5)
            self.label_predictor_layer2 = Dense(10, activation=None)
        elif depth == 4:
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1 = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2 = Dropout(0.5)
            self.label_predictor_layer3 = Dense(10, activation=None)
        elif depth == 5:
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1 = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2 = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3 = Dropout(0.5)
            self.label_predictor_layer4 = Dense(10, activation=None)
        elif depth == 6:    
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1 = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2 = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3 = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4 = Dropout(0.5)
            self.label_predictor_layer5 = Dense(10, activation=None)
        elif depth == 7:
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1 = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2 = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3 = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4 = Dropout(0.5)
            self.label_predictor_layer5 = Dense(100, activation='relu')
            self.train_layer5 = Dropout(0.5)
            self.label_predictor_layer6 = Dense(10, activation=None)
        elif depth == 8:
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1 = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2 = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3 = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4 = Dropout(0.5)
            self.label_predictor_layer5 = Dense(100, activation='relu')
            self.train_layer5 = Dropout(0.5)
            self.label_predictor_layer6 = Dense(100, activation='relu')
            self.train_layer6 = Dropout(0.5)
            self.label_predictor_layer7 = Dense(10, activation=None)
        elif depth == 9:    
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1 = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2 = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3 = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4 = Dropout(0.5)
            self.label_predictor_layer5 = Dense(100, activation='relu')
            self.train_layer5 = Dropout(0.5)
            self.label_predictor_layer6 = Dense(100, activation='relu')
            self.train_layer6 = Dropout(0.5)
            self.label_predictor_layer7 = Dense(100, activation='relu')
            self.train_layer7 = Dropout(0.5)
            self.label_predictor_layer8 = Dense(10, activation=None)
        elif depth ==10:    
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1 = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2 = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3 = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4 = Dropout(0.5)
            self.label_predictor_layer5 = Dense(100, activation='relu')
            self.train_layer5 = Dropout(0.5)
            self.label_predictor_layer6 = Dense(100, activation='relu')
            self.train_layer6 = Dropout(0.5)
            self.label_predictor_layer7 = Dense(100, activation='relu')
            self.train_layer7 = Dropout(0.5)
            self.label_predictor_layer8 = Dense(100, activation='relu')
            self.train_layer8 = Dropout(0.5)
            self.label_predictor_layer9 = Dense(10, activation=None)
        
        #Domain Predictor
        self.domain_predictor_layer0 = GradientReversalLayer()
        self.domain_predictor_layer1 = Dense(100, activation='relu')
        self.domain_predictor_layer2 = Dense(2, activation=None)
        
    def call(self, x, train=False, source_train=True, lamda=1.0):
        '''
            Forward evaluation of input image x through the network
        '''
        #Feature Extractor
        x = self.feature_extractor_layer0(x)
        x = self.feature_extractor_layer1(x, training=train)
        x = self.feature_extractor_layer2(x)
        
        x = self.feature_extractor_layer3(x)
        x = self.feature_extractor_layer4(x, training=train)
        x = self.feature_extractor_layer5(x, training=train)
        x = self.feature_extractor_layer6(x)
        feature = tf.reshape(x, [-1, 4 * 4 * 64])
        
        #Label Predictor
        depth = self.depth
        if source_train is True:
            feature_slice = feature
        else:
            feature_slice = tf.slice(feature, [0, 0], [feature.shape[0] // 2, -1])#input, begin,size(tatsächlich keine index zählung)
            # nur von 1,1024 zu 0,1024?
        #lp_x = self.label_predictor_layer0(feature_slice)
        #lp_x = self.label_predictor_layer1(lp_x)
        #l_logits = self.label_predictor_layer2(lp_x)
        if depth == 1:
            l_logits = self.label_predictor_layer0(feature_slice)
        elif depth == 2:
            lp_x =self.label_predictor_layer0(feature_slice)
            l_logits = self.label_predictor_layer1(lp_x)
        elif depth == 3:    
            lp_x =self.label_predictor_layer0(feature_slice)
            lp_x =self.label_predictor_layer1(lp_x)
            lp_x= self.train_layer1(lp_x, training=train)
            l_logits =self.label_predictor_layer2(lp_x)
        elif depth == 4:
            lp_x =self.label_predictor_layer0(feature_slice)
            lp_x =self.label_predictor_layer1(lp_x)
            lp_x= self.train_layer1(lp_x, training=train)
            lp_x =self.label_predictor_layer2(lp_x)
            lp_x= self.train_layer2(lp_x, training=train)
            l_logits =self.label_predictor_layer3(lp_x)
        elif depth == 5:
            lp_x =self.label_predictor_layer0(feature_slice)
            lp_x =self.label_predictor_layer1(lp_x)
            lp_x= self.train_layer1(lp_x, training=train)
            lp_x =self.label_predictor_layer2(lp_x)
            lp_x= self.train_layer2(lp_x, training=train)
            lp_x =self.label_predictor_layer3(lp_x)
            lp_x= self.train_layer3(lp_x, training=train)
            l_logits =self.label_predictor_layer4(lp_x)
        elif depth == 6:    
            lp_x =self.label_predictor_layer0(feature_slice)
            lp_x =self.label_predictor_layer1(lp_x)
            lp_x= self.train_layer1(lp_x, training=train)
            lp_x =self.label_predictor_layer2(lp_x)
            lp_x= self.train_layer2(lp_x, training=train)
            lp_x =self.label_predictor_layer3(lp_x)
            lp_x= self.train_layer3(lp_x, training=train)
            lp_x =self.label_predictor_layer4(lp_x)
            lp_x= self.train_layer4(lp_x, training=train)
            l_logits =self.label_predictor_layer5(lp_x)
        elif depth == 7:
            lp_x =self.label_predictor_layer0(feature_slice)
            lp_x =self.label_predictor_layer1(lp_x)
            lp_x= self.train_layer1(lp_x, training=train)
            lp_x =self.label_predictor_layer2(lp_x)
            lp_x= self.train_layer2(lp_x, training=train)
            lp_x =self.label_predictor_layer3(lp_x)
            lp_x= self.train_layer3(lp_x, training=train)
            lp_x =self.label_predictor_layer4(lp_x)
            lp_x= self.train_layer4(lp_x, training=train)
            lp_x =self.label_predictor_layer5(lp_x)
            lp_x= self.train_layer5(lp_x, training=train)
            l_logits =self.label_predictor_layer6(lp_x)
        elif depth == 8:
            lp_x =self.label_predictor_layer0(feature_slice)
            lp_x =self.label_predictor_layer1(lp_x)
            lp_x= self.train_layer1(lp_x, training=train)
            lp_x =self.label_predictor_layer2(lp_x)
            lp_x= self.train_layer2(lp_x, training=train)
            lp_x =self.label_predictor_layer3(lp_x)
            lp_x= self.train_layer3(lp_x, training=train)
            lp_x =self.label_predictor_layer4(lp_x)
            lp_x= self.train_layer4(lp_x, training=train)
            lp_x =self.label_predictor_layer5(lp_x)
            lp_x= self.train_layer5(lp_x, training=train)
            lp_x =self.label_predictor_layer6(lp_x)
            lp_x= self.train_layer6(lp_x, training=train)
            l_logits =self.label_predictor_layer7(lp_x)
        elif depth == 9:    
            lp_x =self.label_predictor_layer0(feature_slice)
            lp_x =self.label_predictor_layer1(lp_x)
            lp_x= self.train_layer1(lp_x, training=train)
            lp_x =self.label_predictor_layer2(lp_x)
            lp_x= self.train_layer2(lp_x, training=train)
            lp_x =self.label_predictor_layer3(lp_x)
            lp_x= self.train_layer3(lp_x, training=train)
            lp_x =self.label_predictor_layer4(lp_x)
            lp_x= self.train_layer4(lp_x, training=train)
            lp_x =self.label_predictor_layer5(lp_x)
            lp_x= self.train_layer5(lp_x, training=train)
            lp_x =self.label_predictor_layer6(lp_x)
            lp_x= self.train_layer6(lp_x, training=train)
            lp_x =self.label_predictor_layer7(lp_x)
            lp_x= self.train_layer7(lp_x, training=train)
            l_logits =self.label_predictor_layer8(lp_x)
        elif depth == 10:    
            lp_x =self.label_predictor_layer0(feature_slice)
            lp_x =self.label_predictor_layer1(lp_x)
            lp_x= self.train_layer1(lp_x, training=train)
            lp_x =self.label_predictor_layer2(lp_x)
            lp_x= self.train_layer2(lp_x, training=train)
            lp_x =self.label_predictor_layer3(lp_x)
            lp_x= self.train_layer3(lp_x, training=train)
            lp_x =self.label_predictor_layer4(lp_x)
            lp_x= self.train_layer4(lp_x, training=train)
            lp_x =self.label_predictor_layer5(lp_x)
            lp_x= self.train_layer5(lp_x, training=train)
            lp_x =self.label_predictor_layer6(lp_x)
            lp_x= self.train_layer6(lp_x, training=train)
            lp_x =self.label_predictor_layer7(lp_x)
            lp_x= self.train_layer7(lp_x, training=train)
            lp_x =self.label_predictor_layer8(lp_x)
            lp_x= self.train_layer8(lp_x, training=train)
            l_logits =self.label_predictor_layer9(lp_x)
        #Domain Predictor
        if source_train is True:
            return l_logits
        else:
            dp_x = self.domain_predictor_layer0(feature, lamda)    #GradientReversalLayer
            dp_x = self.domain_predictor_layer1(dp_x)
            d_logits = self.domain_predictor_layer2(dp_x)
            return l_logits, d_logits
        
def loss_func(input_logits, target_labels):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=input_logits, labels=target_labels))
        
def get_loss(l_logits, labels, d_logits=None, domain=None):
            if d_logits is None:
                return loss_func(l_logits, labels)
            else:
                a = loss_func(l_logits, labels)
                b = loss_func(d_logits, domain)
                return a + b

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
        EPOCH = 200
        m = 28
        #%% Create and prepare Data
        mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y         = data_create(source_domain, source_domain, snr = snr_s, plotsamples = plot_source_samples)
        mnist_m_train_x, mnist_m_train_y, mnist_m_test_x, mnist_m_test_y = data_create(target_domain, target_domain, snr = snr_s, plotsamples = plot_target_samples)
            #1D,1D,3D,3D,3D,1D
        Cha = [1,1,3,3,3,1]
        '''Make all data 3-channel images'''
        if DA == True:
            if Cha[source_domain]   <   Cha[target_domain]:
                mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
                mnist_test_x = np.repeat(mnist_test_x, 3, axis=3)
            elif Cha[source_domain] >   Cha[target_domain]:
                mnist_m_train_x = np.repeat(mnist_m_train_x, 3, axis=3)
                mnist_m_test_x = np.repeat(mnist_m_test_x, 3, axis=3)
        else:
            if Cha[source_domain]   <  3:
                mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
                mnist_test_x = np.repeat(mnist_test_x, 3, axis=3)
            if  Cha[target_domain] < 3:
                mnist_m_train_x = np.repeat(mnist_m_train_x, 3, axis=3)
                mnist_m_test_x = np.repeat(mnist_m_test_x, 3, axis=3)
        '''Make data tensorflowable'''
        inputsize = mnist_test_x[1]*mnist_test_x[2]*mnist_test_x[3]
        mnist_train_x   = mnist_train_x.astype('float32')
        mnist_test_x    = mnist_test_x.astype('float32')
        mnist_m_train_x = mnist_m_train_x.astype('float32')
        mnist_m_test_x  = mnist_m_test_x.astype('float32')
        mnist_train_y_dec = mnist_train_y
        mnist_test_y_dec  = mnist_test_y
        mnist_train_y = to_categorical(mnist_train_y)
        mnist_test_y = to_categorical(mnist_test_y)
        mnist_m_train_y_dec = mnist_m_train_y
        mnist_m_test_y_dec  = mnist_m_test_y
        mnist_m_train_y = to_categorical(mnist_m_train_y)
        mnist_m_test_y = to_categorical(mnist_m_test_y)
        '''Create Batches'''
        source_dataset = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y)).shuffle(1000).batch(BATCH_SIZE*2)
        da_dataset = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y, mnist_m_train_x, mnist_m_train_y)).shuffle(1000).batch(BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices((mnist_m_test_x, mnist_m_test_y)).shuffle(1000).batch(BATCH_SIZE*2) #Test Dataset over Target Domain
        test_dataset2 = tf.data.Dataset.from_tensor_slices((mnist_m_train_x, mnist_m_train_y)).shuffle(1000).batch(BATCH_SIZE*2) #Test Dataset over Target (used for training)
        domain_labels = np.vstack([np.tile([1., 0.], [BATCH_SIZE, 1]),np.tile([0., 1.], [BATCH_SIZE, 1])])
        domain_labels = domain_labels.astype('float32')
        '''Training'''
        model = DANN(depth = depth+1)
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        source_acc = []  # Source Domain Accuracy while Source-only Training
        da_acc = []      # Source Domain Accuracy while DA-training
        test_acc = []    # Testing Dataset (Target Domain) Accuracy 
        test2_acc = []   # Target Domain (used for Training) Accuracy
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
        if safe_weights:
            if DA:
                model.save_weights('trained_networks_SourceOnly\DANN_S' + str(source_domain) +'_T' + str(target_domain) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'da.h5')
            else:
                model.save_weights('trained_networks_DomAdapt\DANN_S' + str(source_domain) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'source.h5')
        print("iteration done")
        print("depth = {}, source = {}, target = {}".format(depth, source_domain, target_domain))