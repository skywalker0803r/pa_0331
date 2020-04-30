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
from PyQt5 import QtCore, QtGui, QtWidgets

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

class pa_api(object):
    def __init__(self):
        self.tag_map = tag_map
        self.critic = critic.eval()
        self.actor = net.eval()
        self.mm_x = mm_x
        self.mm_y = mm_y
        self.x_col = x_cols
        self.y_cols = y_cols
        self.time_step = t
        self.num_sensor = n
    
    def get_advice(self,set_point):
        st = self.mm_y.transform(np.array([[set_point]]))
        st = torch.tensor(st).cuda()
        return self.actor(st)

    def get_critic_output(self,advice):
        output = self.critic(advice).detach().cpu().numpy()
        return self.mm_y.inverse_transform(output)
    
    def pretty_advice(self,advice):
        advice = advice.detach().cpu().numpy()
        advice = self.mm_x.inverse_transform(advice)
        advice = advice.reshape(self.time_step,self.num_sensor)
        advice = pd.DataFrame(advice,columns=self.x_col)
        advice = advice.describe().T[['50%','min','max']]
        advice['chinese'] = advice.index.map(self.tag_map)
        return advice[['chinese','50%','min','max']]

class PandasModel(QtCore.QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None): 
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df.copy()

    def toDataFrame(self):
        return self._df.copy()

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()