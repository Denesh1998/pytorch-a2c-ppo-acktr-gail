#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:51:26 2021

@author: denesh
"""
import pickle
import pandas as pd
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy import convolve
import matplotlib.pyplot as plt
 
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
 

#exchange_rates = pd.read_csv("rewards_avg.txt")
# path = Path('~/Acads/Q3/DL/pytorch-a2c-ppo-acktr-gail/save_data/').expanduser()

# rl_mean= torch.load(path/'torch_db')

# print(rl_mean['rl'])      
with open("save_data/rewards_3.txt", "rb") as fp:   # Unpickling
    rl_mean = pickle.load(fp)

    
rl_mean  = np.array(rl_mean)
print(np.shape(rl_mean))
x= np.arange(100,4000100,100)
yMA = movingaverage(rl_mean,1000)
#print yMA
plt.figure(2)
plt.plot(x[len(x)-len(yMA):],yMA)
plt.show()