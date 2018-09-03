#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:00:17 2018

@author: thinkpad
"""
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
from base.nn import BaseNN, normc_initializer

class DQNet(BaseNN):
    name="DQNet"
    
    def __init__(self, input_shape, output_shape):
        super(DQNet, self).__init__(input_shape, output_shape)
        self.conv = nn.Sequential(nn.Conv2d(self.input_shape[0], 8, kernel_size=8, stride=4), nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=4, stride=2), nn.Tanh(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.Tanh())
        self.dense = nn.Sequential(nn.Linear(self.linear_output_shape(self.conv), 512),nn.Tanh(),
                                    nn.Linear(512,output_shape))
        
    def forward(self, x):
        y = self.conv(x)
        return self.dense(y.view(y.size(0),-1))
    def compile(self):
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025,alpha=0.95,eps=0.01,momentum=0.95)



        
class TRPONet(DQNet):
    name="TRPONet"
    def compile(self):
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(),lr=1e-5,alpha=0.99,weight_decay=1e-4)

class TRPONetValue(DQNet):
    name="TRPONetValue"
    def __init__(self, input_shape, output_shape):
        super(TRPONetValue, self).__init__(input_shape, 1)




class DQNetS(BaseNN):
    name="DQNet_S"
    
    def __init__(self, input_shape, output_shape):
        super(DQNetS, self).__init__(input_shape, output_shape)
        self.conv = nn.Sequential(nn.Conv2d(self.input_shape[0], 16, kernel_size=8, stride=4), nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU())
        self.dense = nn.Sequential(nn.Linear(self.linear_output_shape(self.conv), 256),nn.ReLU(),
                                    nn.Linear(256,output_shape))
        
    def forward(self, x):
        y = self.conv(x)
        return self.dense(y.view(y.size(0),-1))
    
    def compile(self):
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025,alpha=0.95,eps=0.01,momentum=0.95)

class TRPONetS(DQNetS):
    name="TRPONet_S"

class TRPONetSValue(TRPONetS):
    name="TRPONet_S_Value"
    def __init__(self, input_shape, output_shape):
        super(TRPONetSValue, self).__init__(input_shape, 1)
