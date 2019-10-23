# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:28:37 2019

@author: lth
"""

layer1 = g[0]
layer2 = g[1]
layer3 = g[2]

def sigmoid(x):

        return 1 / (1 + math.exp(-x))
 
def relu(x):
    return np.maximum(x,0)

def network(x):
    return sigmoid(relu( relu(x.dot(g[12])).dot(g[13])).dot(g[14]))


res = [network(np.array(i)) for i in np.linspace(80,120,80)]