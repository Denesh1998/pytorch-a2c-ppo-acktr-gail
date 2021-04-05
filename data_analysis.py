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
    rg1 = pickle.load(fp)

with open("save_data/GRU/rewards_2_v1.txt", "rb") as fp:   # Unpickling
    rg2 = pickle.load(fp)

with open("save_data/GRU/rewards_3_v1.txt", "rb") as fp:   # Unpickling
    rg3 = pickle.load(fp)
with open("save_data/GRU/rewards_3_p1.txt", "rb") as fp:   # Unpickling
    rp3 = pickle.load(fp)
with open("save_data/IAM/rewards_1_v1.txt", "rb") as fp:   # Unpickling
    rI1 = pickle.load(fp)

with open("save_data/IAM/rewards_2_v1.txt", "rb") as fp:   # Unpickling
    rI2 = pickle.load(fp)

with open("save_data/IAM/rewards_3_v1.txt", "rb") as fp:   # Unpickling
    rI3 = pickle.load(fp)

    
rg1  = np.array(rg1)
rg2  = np.array(rg2)
rg3  = np.array(rg3)
rp3  = np.array(rp3)
rI1  = np.array(rI1)
rI2  = np.array(rI2)
rI3  = np.array(rI3)
x= np.arange(10100,4000100,100)
rg1MA = movingaverage(rg1,1000).reshape(1,-1)
rg2MA = movingaverage(rg2,1000).reshape(1,-1)
rg3MA = movingaverage(rg3,1000).reshape(1,-1)
rp3MA = movingaverage(rp3,1000)
rI1MA = movingaverage(rI1,1000).reshape(1,-1)
rI2MA = movingaverage(rI2,1000).reshape(1,-1)
rI3MA = movingaverage(rI3,1000).reshape(1,-1)

rg = np.vstack((rg1MA, rg2MA,rg3MA))
rgl_mean =  np.mean(rg,axis=0)
rgl_std = np.std(rg,axis=0)

rI = np.vstack((rI1MA, rI2MA,rI3MA))
rIl_mean =  np.mean(rI,axis=0)
rIl_std = np.std(rI,axis=0)


plt.fill_between(x[len(x)-rIl_mean.shape[0]:], rIl_mean - rIl_std, rIl_mean + rIl_std, alpha=0.5) 
plt.fill_between(x[len(x)-rgl_mean.shape[0]:], rgl_mean - rgl_std, rgl_mean + rgl_std, alpha=0.5)   
# plt.plot(rl_mean)   
# plt.show()
#print yMA
plt.xlabel("Steps")
plt.xlim(0,4e6)
plt.ylim((26,42))
plt.ylabel("Average reward")
plt.title("Average Performance on Warehouse")
plt.plot(x[len(x)-rIl_mean.shape[0]:],rIl_mean)
plt.plot(x[len(x)-rgl_mean.shape[0]:],rgl_mean)
# plt.plot(x[len(x)-rl_mean.shape[0]:],rp3MA)
plt.legend(('IAM', 'GRU'))
plt.show()