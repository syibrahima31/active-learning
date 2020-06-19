#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:22:19 2020

@author: ibrahima
"""


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from keras import models
from keras import layers 
from keras.utils import np_utils




#import the dataset mnist 

mnist = keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = (x_train.reshape((x_train.shape[0], 28*28)).astype('float32')/ 255)[0:1000,:]

Y_train = (np_utils.to_categorical(y_train, 10))[0:1000]



# define model CNN 

# def ConlutionalNetwork():
#     model = keras.Model()
#     model.keras.layers 
y_train =y_train[0:1000].reshape((1000,1))

data = np.concatenate((x_train, y_train), axis=1)


base = np.copy(data)


def LogisticRegression():
    
    model = models.Sequential()
    
    model.add(layers.Dense(100, input_shape=(28*28,), activation= "relu"))
    #model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))
    
    
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

    return model 




def active_learning(data, n_iter, n_sample, epochs, batch_size):
    """
    The training dataset is increased by n_sample example at every iteration.
    Args:
        data: Pool of unseen data
        n_iter: Int. Number of iteration to perform the active learning procedure
        n_sample: Int. Number of sample per iteration
    Returns:
        evaluation: List of float. The evaluation of the model trained on data
        training_data: Total data we have trained on
        weights: parameters of the model at each iteration
    """
    evaluation = []
    weights = []
    for i in range(n_iter):
        print("Iteration: {}".format(i+1))        
        if i == 0:
            sampled_data, data = sample_random(n_sample,data)
            training_data = sampled_data  
            
            
        model =  LogisticRegression()
        print("Start training")
        X = training_data[:,:784]
        Y = np_utils.to_categorical(training_data[:,-1], 10)
        model.fit(X, Y, epochs=epochs, batch_size=batch_size ,verbose=0, shuffle=True)
        print("End training")
        Data_X = data[:,:784]
        Data_Y  = np_utils.to_categorical(data[:,-1],10)
        eval_i = model.evaluate( Data_X, Data_Y)[1]
        evaluation.append(eval_i)
        print("Accuracy: {}".format(eval_i))        
        weights.append(model.get_weights())
        sampled_data, rest_data = mnist_least_confidence(model, data, n_sample)
        data = rest_data
#        import pdb; pdb.set_trace()
        training_data = np.concatenate((training_data, sampled_data), axis =0)        
        print("---------------------------")
    return evaluation, weights, training_data



evaluation_lsm, weights_ls, training_ls  = active_learning(base, n_iter=10, n_sample=100, epochs=10,batch_size=10)


plt.plot(evaluation_lsm)
