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

#exchange_rates = pd.read_csv("rewards_avg.txt")
# path = Path('~/Acads/Q3/DL/pytorch-a2c-ppo-acktr-gail/save_data/').expanduser()

# rl_mean= torch.load(path/'torch_db')

# print(rl_mean['rl'])      
with open("save_data/rewards_2.txt", "rb") as fp:   # Unpickling
    rl_mean = pickle.load(fp)

    
rl_mean  = np.array(rl_mean)
print(np.shape(rl_mean))
steps = np.arange(100,4100,100)
plt.plot(steps,rl_mean)