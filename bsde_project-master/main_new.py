# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:48:30 2019

@author: lth
"""
import matplotlib.pyplot as plt
from equation import EuropeanCall
from config import EuropeanCallConfig
from solver import FeedForwardModel
import tensorflow.compat.v1 as tf
import numpy as np
from scipy.stats import norm

def blackscholes_price(K, T, S, vol, r=0, q=0, callput='call'):
    F = S*np.exp((r-q)*T)
    v = np.sqrt(vol**2*T)
    d1 = np.log(F/K)/v + 0.5*v
    d2 = d1 - v
    try:
        opttype = {'call':1, 'put':-1}[callput.lower()]
    except:
        raise ValueError('The value of callput must be either "call" or "put".')
    price = opttype*(F*norm.cdf(opttype*d1)-K*norm.cdf(opttype*d2))*np.exp(-r*T)
    return price



#params
dim, total_time, num_time_interval = 1, 1, 10
sigma, r, K = 0.2, 0.02, 100
ob_range = (80, 120)

#fit
eurOption= EuropeanCall(dim, total_time, num_time_interval, sigma, r, K, ob_range)
tf.reset_default_graph()
with tf.Session() as sess:
    model = FeedForwardModel(sess, eurOption, EuropeanCallConfig())
    model.build()
    f_graphs,z_graphs= model.train()

#plot
fig, ax = plt.subplots()
l, u = ob_range
num = EuropeanCallConfig().ob_num
ax.plot(np.linspace(l, u, num), f_graphs.flatten())
ax.plot(np.linspace(l, u, num), np.maximum(np.linspace(l, u, num) - K, 0), 'r')
ax.plot(np.linspace(l, u, num), blackscholes_price(K, total_time, np.linspace(l, u, num), sigma, r, 0, 'call'),'g')

for i in range(len(z_graphs)):
    fig, ax = plt.subplots()
    ax.plot(np.linspace(l, u, num),np.array(z_graphs[i]).flatten())