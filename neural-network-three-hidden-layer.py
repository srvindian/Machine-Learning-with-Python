# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:45:03 2019

@author: Sourav
"""
#odd even count with four layer  (3-in)---(4-hidden1)---(4-hidden2)--(4-hiddden3)--(1-op)
# one bias unit for each input layer and hidden1 layer

import numpy as np
#%%
def predict(x, th0, th1, th2):
    l1 = 1 / (1 + np.exp( - (np.dot(x, th0))))
    l2 = 1 / (1 + np.exp( - (np.dot(l1, th1))))
    l3 = 1 / (1 + np.exp( - (np.dot(l2, th2))))
    return l3
#%%
if __name__ == "__main__":
    X = np.array([ [0, 0, 1], [0,1,1], [1,0,1], [1,1,1] ])
    y = np.array([ [0, 1, 1, 0] ]).T
    np.random.seed(50)  # seeding is good to stick on same everytim
    theta0 = 2 * np.random.random((3, 4)) - 1
    theta1 = 2 * np.random.random((4, 4)) - 1
    theta2 = 2 * np.random.random((4, 1)) - 1
    
    for j in range(0,6000):
        #feedforward
        l1 = 1 / (1 + np.exp( - (np.dot(X, theta0))))
        l2 = 1 / (1 + np.exp( - (np.dot(l1, theta1))))
        l3 = 1 / (1 + np.exp( - (np.dot(l2, theta2))))
        #backpropagation
        l3_delta = (y - l3) * (l3 * (1 - l3))
        l2_delta = l3_delta.dot(theta2.T) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(theta1.T) * (l1 * (1 - l1))
        theta2 += l2.T.dot(l3_delta)
        theta1 += l1.T.dot(l2_delta)
        theta0 += X.T.dot(l1_delta)
    #%% testing
    res = predict(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]), theta0, theta1, theta2)
    print(res.round())
    
