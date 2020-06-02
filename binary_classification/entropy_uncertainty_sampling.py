#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:23:04 2020

@author: jmamath
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Dense, Input
#from utils import sample_random, plot_decision_boundary, sample_highest_entropy


#################### INITIAL DATA #################### 
# We start by generating data from two normal distribution on the plane.
x1 = np.random.multivariate_normal(mean=np.array([-2,0]), cov=np.eye(2), size = 200)
x2 = np.random.multivariate_normal(mean=np.array([2,0]), cov=np.eye(2), size = 200)

# We give to each data belonging a label.
y1 = np.zeros([200,1])
y2 = np.ones([200,1])


x1 = np.concatenate([x1,y1], axis=-1)
x2 = np.concatenate([x2,y2], axis=-1)
x = np.concatenate([x1,x2]).astype(np.float32)


# Display the original data
plt.scatter(x1[:,0], x1[:,1], s=10., label='N1')
plt.scatter(x2[:,0], x2[:,1], s=10., label='N2')
plt.legend()

# Now, we duplicate in two the dataset to use in each procedure
# random sampling with data_rs, and uncertainty sampling with data_us
data_us = np.copy(x)

# Now we train a logistic regression clasifier on the data previously sampled
def logistic_regression():
    x_ = Input(shape=(2,))    
    out = Dense(1, activation='sigmoid', activity_regularizer=l2())(x_)
    model = Model(x_, out)
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    return model

####################  ACTIVE LEARNING #################### 
# Ok let's start with the classifier at hand, now we want to query 
# 10 examples with the highest entropy
# and sample 10 more examples at random
def active_learning(data, n_iter, n_sample, epochs):
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
        model = logistic_regression()
        print("Start training")
        model.fit(training_data[:,:2], training_data[:,2], epochs=epochs, verbose=0, shuffle=True)
        print("End training")
        eval_i = model.evaluate(data[:,:2], data[:,2])[1]
        evaluation.append(eval_i)
        print("Accuracy: {}".format(eval_i))        
        weights.append(model.get_weights())
        sampled_data, data = sample_highest_entropy(n_sample, model, data)
#        import pdb; pdb.set_trace()
        training_data = np.concatenate([training_data, sampled_data])        
        print("---------------------------")
    return evaluation, weights, training_data


evaluation_us, weights_us, training_us  = active_learning(data_us, n_iter=39, n_sample=10, epochs=500)


line_us_x, line_us_y = plot_decision_boundary(weights_us)

# Plot the decision boundary learned with the uniformly sampled data
id0 = training_us[:,2] == 0
id1 = training_us[:,2] == 1
rs_id0 = training_us[id0]
rs_id1 = training_us[id1]

# Start with just the data we trained with
plt.scatter(rs_id0[:,0], rs_id0[:,1], s=30., label='N1')
plt.scatter(rs_id1[:,0], rs_id1[:,1], s=30., label='N2')
plt.plot(line_us_x, line_us_y, label="Uncertainty Sampling")
plt.title("Decision Boundary with the data we trained on ")
plt.legend()

# Plot the decition boundary with all the data
plt.scatter(x1[:,0], x1[:,1], s=10., label='N1')
plt.scatter(x2[:,0], x2[:,1], s=10., label='N2')
plt.plot(line_us_x, line_us_y, label="Uncertainty Sampling")
plt.title("Decision Boundary with Total Data")
plt.legend()




