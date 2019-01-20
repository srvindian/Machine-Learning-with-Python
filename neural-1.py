# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:45:03 2019

@author: Sourav
"""
#odd even count with three layer  (3-in)---(3-hidden1)---(3-hidden2)--(1-op)
# one bias unit for each input layer and hidden1 layer

import numpy as np
#%%
def predict(x, th0, th1):
    l1 = 1 / (1 + np.exp( - (np.dot(x, th0))))
    l2 = 1 / (1 + np.exp( - (np.dot(l1, th1))))
    return l2
#%%
if __name__ == "__main__":
    X = np.array([ [0, 0, 1], [0,1,1], [1,0,1], [1,1,1] ])
    y = np.array([ [0, 1, 1, 0] ]).T
    np.random.seed(50)  # seeding is good to stick on same everytim
    theta0 = 2 * np.random.random((3, 4)) - 1
    theta1 = 2 * np.random.random((4, 1)) - 1
    for j in range(0,6000):
        #feedforward
        l1 = 1 / (1 + np.exp( - (np.dot(X, theta0))))
        l2 = 1 / (1 + np.exp( - (np.dot(l1, theta1))))
        #backpropagation
        l2_delta = (y - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(theta1.T) * (l1 * (1 - l1))
        theta1 += l1.T.dot(l2_delta)
        theta0 += X.T.dot(l1_delta)
    #%% testing
    res = predict(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]), theta0, theta1)
    print(res.round())
    