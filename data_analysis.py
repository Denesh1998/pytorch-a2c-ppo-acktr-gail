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
with open("save_data/GRU/rewards_1_v1.txt", "rb") as fp:   # Unpickling
    r1 = pickle.load(fp)

with open("save_data/GRU/rewards_2_v1.txt", "rb") as fp:   # Unpickling
    r2 = pickle.load(fp)

with open("save_data/GRU/rewards_3_v1.txt", "rb") as fp:   # Unpickling
    r3 = pickle.load(fp)

    
r1  = np.array(r1)
r2  = np.array(r2)
r3  = np.array(r3)
x= np.arange(10100,4000100,100)
r1MA = movingaverage(r1,1000).reshape(1,-1)
r2MA = movingaverage(r2,1000).reshape(1,-1)
r3MA = movingaverage(r3,1000).reshape(1,-1)

r = np.vstack((r1MA, r2MA,r3MA))
rl_mean =  np.mean(r,axis=0)

rl_std = np.std(r,axis=0)

plt.fill_between(x[len(x)-rl_mean.shape[0]:], rl_mean - rl_std, rl_mean + rl_std, alpha=0.5)  
# plt.plot(rl_mean)   
# plt.show()
#print yMA
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.title("GRU Performance on Warehouse")
plt.plot(x[len(x)-rl_mean.shape[0]:],rl_mean)
plt.show()