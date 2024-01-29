# -*- coding: utf-8 -*-
"""
Define The DANN Network in Tensorflow

Created on Thu Jan  4 12:42:39 2024

@author: savci
"""
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras import  Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Dropout

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
    Initialize a DANN Model with feature extraction and different depth label classification networks
    TODO: Depth n initializes network of depth 3 + regularization. More shallow networks need to be hardcoded right now
    CHECK: Can vectorize initialization with naming conventions of layers?
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
            self.train_layer1           = Dropout(0.5)
            self.label_predictor_layer2 = Dense(10, activation=None)
        elif depth == 4:
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1           = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2           = Dropout(0.5)
            self.label_predictor_layer3 = Dense(10, activation=None)
        elif depth == 5:
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1           = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2           = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3           = Dropout(0.5)
            self.label_predictor_layer4 = Dense(10, activation=None)
        elif depth == 6:    
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1           = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2           = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3           = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4           = Dropout(0.5)
            self.label_predictor_layer5 = Dense(10, activation=None)
        elif depth == 7:
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1           = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2           = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3           = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4           = Dropout(0.5)
            self.label_predictor_layer5 = Dense(100, activation='relu')
            self.train_layer5           = Dropout(0.5)
            self.label_predictor_layer6 = Dense(10, activation=None)
        elif depth == 8:    
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1           = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2           = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3           = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4           = Dropout(0.5)
            self.label_predictor_layer5 = Dense(100, activation='relu')
            self.train_layer5           = Dropout(0.5)
            self.label_predictor_layer6 = Dense(100, activation='relu')
            self.train_layer6           = Dropout(0.5)
            self.label_predictor_layer7 = Dense(10, activation=None)
        elif depth == 9:    
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1           = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2           = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3           = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4           = Dropout(0.5)
            self.label_predictor_layer5 = Dense(100, activation='relu')
            self.train_layer5           = Dropout(0.5)
            self.label_predictor_layer6 = Dense(100, activation='relu')
            self.train_layer6           = Dropout(0.5)
            self.label_predictor_layer7 = Dense(100, activation='relu')
            self.train_layer7           = Dropout(0.5)
            self.label_predictor_layer8 = Dense(10, activation=None)
        elif depth ==10:        
            self.label_predictor_layer0 = Dense(100, activation='relu')
            self.label_predictor_layer1 = Dense(100, activation='relu')
            self.train_layer1           = Dropout(0.5)
            self.label_predictor_layer2 = Dense(100, activation='relu')
            self.train_layer2           = Dropout(0.5)
            self.label_predictor_layer3 = Dense(100, activation='relu')
            self.train_layer3           = Dropout(0.5)
            self.label_predictor_layer4 = Dense(100, activation='relu')
            self.train_layer4           = Dropout(0.5)
            self.label_predictor_layer5 = Dense(100, activation='relu')
            self.train_layer5           = Dropout(0.5)
            self.label_predictor_layer6 = Dense(100, activation='relu')
            self.train_layer6           = Dropout(0.5)
            self.label_predictor_layer7 = Dense(100, activation='relu')
            self.train_layer7           = Dropout(0.5)
            self.label_predictor_layer8 = Dense(100, activation='relu')
            self.train_layer8           = Dropout(0.5)
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
        if depth == 1:
            l_logits    = self.label_predictor_layer0(feature_slice)
        elif depth == 2:
            rep         = self.label_predictor_layer0(feature_slice)
            l_logits    = self.label_predictor_layer1(rep)
        elif depth == 3:    
            rep         = self.label_predictor_layer0(feature_slice)
            rep         = self.label_predictor_layer1(rep)
            rep         = self.train_layer1(rep, training=train)
            l_logits    = self.label_predictor_layer2(rep)
        elif depth == 4:
            rep        = self.label_predictor_layer0(feature_slice)
            rep        = self.label_predictor_layer1(rep)
            rep        = self.train_layer1(rep, training=train)
            rep        = self.label_predictor_layer2(rep)
            rep        = self.train_layer2(rep, training=train)
            l_logits    = self.label_predictor_layer3(rep)
        elif depth == 5:
            rep        = self.label_predictor_layer0(feature_slice)
            rep        = self.label_predictor_layer1(rep)
            rep        = self.train_layer1(rep, training=train)
            rep        = self.label_predictor_layer2(rep)
            rep        = self.train_layer2(rep, training=train)
            rep        = self.label_predictor_layer3(rep)
            rep        = self.train_layer3(rep, training=train)
            l_logits    = self.label_predictor_layer4(rep)
        elif depth == 6:    
            rep        = self.label_predictor_layer0(feature_slice)
            rep        = self.label_predictor_layer1(rep)
            rep        = self.train_layer1(rep, training=train)
            rep        = self.label_predictor_layer2(rep)
            rep        = self.train_layer2(rep, training=train)
            rep        = self.label_predictor_layer3(rep)
            rep        = self.train_layer3(rep, training=train)
            rep        = self.label_predictor_layer4(rep)
            rep        = self.train_layer4(rep, training=train)
            l_logits    = self.label_predictor_layer5(rep)
        elif depth == 7:
            rep        = self.label_predictor_layer0(feature_slice)
            rep        = self.label_predictor_layer1(rep)
            rep        = self.train_layer1(rep, training=train)
            rep        = self.label_predictor_layer2(rep)
            rep        = self.train_layer2(rep, training=train)
            rep        = self.label_predictor_layer3(rep)
            rep        = self.train_layer3(rep, training=train)
            rep        = self.label_predictor_layer4(rep)
            rep        = self.train_layer4(rep, training=train)
            rep        = self.label_predictor_layer5(rep)
            rep        = self.train_layer5(rep, training=train)
            l_logits    = self.label_predictor_layer6(rep)
        elif depth == 8:
            rep        = self.label_predictor_layer0(feature_slice)
            rep        = self.label_predictor_layer1(rep)
            rep        = self.train_layer1(rep, training=train)
            rep        = self.label_predictor_layer2(rep)
            rep        = self.train_layer2(rep, training=train)
            rep        = self.label_predictor_layer3(rep)
            rep        = self.train_layer3(rep, training=train)
            rep        = self.label_predictor_layer4(rep)
            rep        = self.train_layer4(rep, training=train)
            rep        = self.label_predictor_layer5(rep)
            rep        = self.train_layer5(rep, training=train)
            rep        = self.label_predictor_layer6(rep)
            rep        = self.train_layer6(rep, training=train)
            l_logits    = self.label_predictor_layer7(rep)
        elif depth == 9:    
            rep        = self.label_predictor_layer0(feature_slice)
            rep        = self.label_predictor_layer1(rep)
            rep        = self.train_layer1(rep, training=train)
            rep        = self.label_predictor_layer2(rep)
            rep        = self.train_layer2(rep, training=train)
            rep        = self.label_predictor_layer3(rep)
            rep        = self.train_layer3(rep, training=train)
            rep        = self.label_predictor_layer4(rep)
            rep        = self.train_layer4(rep, training=train)
            rep        = self.label_predictor_layer5(rep)
            rep        = self.train_layer5(rep, training=train)
            rep        = self.label_predictor_layer6(rep)
            rep        = self.train_layer6(rep, training=train)
            rep        = self.label_predictor_layer7(rep)
            rep        = self.train_layer7(rep, training=train)
            l_logits    = self.label_predictor_layer8(rep)
        elif depth == 10:    
            rep        = self.label_predictor_layer0(feature_slice)
            rep        = self.label_predictor_layer1(rep)
            rep        = self.train_layer1(rep, training=train)
            rep        = self.label_predictor_layer2(rep)
            rep        = self.train_layer2(rep, training=train)
            rep        = self.label_predictor_layer3(rep)
            rep        = self.train_layer3(rep, training=train)
            rep        = self.label_predictor_layer4(rep)
            rep        = self.train_layer4(rep, training=train)
            rep        = self.label_predictor_layer5(rep)
            rep        = self.train_layer5(rep, training=train)
            rep        = self.label_predictor_layer6(rep)
            rep        = self.train_layer6(rep, training=train)
            rep        = self.label_predictor_layer7(rep)
            rep        = self.train_layer7(rep, training=train)
            rep        = self.label_predictor_layer8(rep)
            rep        = self.train_layer8(rep, training=train)
            l_logits    = self.label_predictor_layer9(rep)
        #Domain Predictor
        if source_train is True:
            return l_logits
        else:
            dom_rep     = self.domain_predictor_layer0(feature, lamda)    #GradientReversalLayer
            dom_rep     = self.domain_predictor_layer1(dom_rep)
            d_logits    = self.domain_predictor_layer2(dom_rep)
            return l_logits, d_logits
        
def loss_func(prediction, labels):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        
def get_loss(l_logits, labels, d_logits=None, domain=None):
            if d_logits is None:
                return loss_func(l_logits, labels)
            else:
                a = loss_func(l_logits, labels)
                b = loss_func(d_logits, domain)
                return a + b
epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
def test_step(model, test_X, test_y):
                    pred_y = model(test_X, train=False, source_train=True)
                    epoch_accuracy(pred_y, test_y)
