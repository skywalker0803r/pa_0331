import pandas as pd
import numpy as np
import torchviz
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import joblib
from sklearn.preprocessing import StandardScaler


class panet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_step = time_step
        self.num_sensor = num_sensor
        self.linear_1 = nn.Linear(self.time_step,1)
        self.block_1 = nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Linear(128,1))
        self.conv_1 = nn.Conv1d(in_channels = self.num_sensor-1,
                                out_channels = 1, 
                                kernel_size = 1,
                                padding = 0)
        
        self.pool = nn.MaxPool1d(kernel_size = 1)
        
        self.block_2 = nn.Sequential(nn.Linear(39,128),nn.ReLU(),nn.Linear(128,1))
        
    def forward(self,x):
        feed,factor = self.fetch(x)
        
        z1 = self.linear_1(feed)
        
        z2 = self.block_1(z1)
        
        z3 = self.pool(F.relu(self.conv_1(factor)))
        z3 = z3.view(-1,z3.shape[1]*z3.shape[2])
        z3 = self.block_2(z3)
        
        return F.sigmoid(z1+z2+z3)
        
    def fetch(self,x):
        x_resize = x.view(-1,self.time_step,self.num_sensor)
        feed = x_resize[:,:,0]
        
        # batch,time,sensor -> batch,sensor,time 
        factor = x_resize[:,:,1:].permute(0,2,1) 
        
        return feed,factor