import pandas as pd
import numpy as np
import torchviz
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import joblib
from sklearn.preprocessing import StandardScaler

class Net(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Net,self).__init__()
        # fc_net
        self.fc = nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,output_shape))

    def forward(self, x):
        return self.fc(x)