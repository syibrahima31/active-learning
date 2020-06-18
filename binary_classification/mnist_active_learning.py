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
x_train = x_train.reshape((x_train.shape[0], 28*28)).astype('float32')/ 255
x_test = x_test.reshape((x_test.shape[0], 28*28)).astype('float32')/ 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10 )


# define model CNN 

# def ConlutionalNetwork():
#     model = keras.Model()
#     model.keras.layers 

x_base = np.concatenate((x_train, x_test), axis=0)
y_base = np.concatenate((y_train, y_test), axis=0).reshape((70_000,1))

data = np.concatenate((x_base, y_base), axis=1)


base = np.copy(data)


def ConvolutionalNetwork():
    
    model = models.Sequential()
    
    model.add(layers.Dense(20, input_shape=(28*28,), activation= "relu"))
    model.add(layers.Dense(20, activation="relu"))
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
            
            
        model = ConvolutionalNetwork()
        print("Start training")
        model.fit(training_data[:,:784], np_utils.to_categorical(training_data[:,-1], 10), epochs=epochs, batch_size=batch_size ,verbose=0, shuffle=True)
        print("End training")
        eval_i = model.evaluate(data[:,:784], np_utils.to_categorical(data[:,-1]),10)[1]
        evaluation.append(eval_i)
        print("Accuracy: {}".format(eval_i))        
        weights.append(model.get_weights())
        sampled_data, rest_data = mnist_least_confidence(model, data)
        data = rest_data
#        import pdb; pdb.set_trace()
        training_data = np.concatenate((training_data, sampled_data), axis =0)        
        print("---------------------------")
    return evaluation, weights, training_data



evaluation_ls, weights_ls, training_ls  = active_learning(base, n_iter=39, n_sample=30000, epochs=10,batch_size=255)




plt.plot(evaluation_ls)











model.fit(x_train,Y_train ,  batch_size = 255, epochs=10 )


prediction = model.predict_proba(x_test)



    
    prediction[:,0]*4

