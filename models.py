#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras model definitions

@author: Baris Bozkurt
"""
import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

#8

#Activation functions => 'relu' or 'sigmoid'
#Activation_functions = 'relu' 
#Activation_functions = 'sigmoid' 

#Padding => 'same' or 'valid'
Padding = 'same' 
#Padding = 'valid' 

#Pooling => MaxPooling2D or AveragePooling2D
#Pooling = MaxPooling2D(pool_size=(2, 2)) 
#Pooling = AveragePooling2D(pool_size=(2, 2))

#Learning rate => 0.01 or 0.001
Learning_rate = 0.01 
#Learning_rate = 0.001 

#Optimizers => Adam or SGD
Optimizers = keras.optimizers.Adam(lr=Learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#Optimizers = keras.optimizers.SGD(lr=Learning_rate, decay=1e-6, momentum=0.9, nesterov=True)


#Strides = (1,1) or (2,2) or (3,3)
#Activation_functions = 'softsign' or 'tanh' or 'relu'
#Pooling = MaxPooling2D(pool_size=(1, 1))  or MaxPooling2D(pool_size=(2, 2))  or MaxPooling2D(pool_size=(3, 3)) 
#Dropout_Rate = 0.1 or 0.2 or 0.3

def loadModel(modelName, input_shape, num_classes,Strides,Activation_functions,Pooling,Dropout_Rate):
    '''Loading one of the Keras models defined below(within this function)

    To specify a new model, please consider adding a new
        elif modelName=="..." block and place the specification within that block

    Args:
        name (str): Name of the model. Ex: 'uocSeq0','uocSeq1', etc.
        input_shape (tuple): Shape of the input vector as expected in definition
            of the input layer of a Keras model
            For spectrogram-like features the following shape info is expected:
            (timeDimension, frequencyDimension, 1)
        num_classes (int): Number of classes
        The last two arguments are used to define the sizes of input and output
            layers of Keras models
    '''
    model=None
    if modelName=="uocSeq1":#Model with two convolutional layers
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), strides=Strides, padding=Padding,activation=Activation_functions, input_shape=input_shape))
        model.add(Conv2D(16, (3, 3), strides=Strides, padding=Padding, activation=Activation_functions))
        model.add(Pooling)
        model.add(Dropout(Dropout_Rate))
        model.add(Flatten()) #512
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(Dropout_Rate))
        model.add(Dense(16,kernel_regularizer=regularizers.l1(0.0002)))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=Optimizers,
                      metrics=['accuracy'])#learning rate default: 0.001

    return model
