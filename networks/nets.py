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
import numpy as np
import core.utils as U
import base.basenetwork as B
from base.basefunction import Policy

class QFunction(B.BaseNetwork):
    name="QFunction"
    
    def __init__(self,env):
        super(QFunction, self).__init__(env.observation_space.shape,env.action_space.n, owner_name=self.name)
        self.conv = [nn.Sequential(nn.Conv2d(self.input_shape[0], 8, kernel_size=8, stride=4), nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=4, stride=2), nn.Tanh(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.Tanh())]
        x = B.output_shape(self.conv[0],self.input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   B.Flatten(),
                                   nn.Linear(np.prod(x), 512),nn.Tanh(),
                                   nn.Linear(512,self.output_shape))

        self.compile()
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025,alpha=0.95,eps=0.01,momentum=0.95)
        

class TRPOPolicy(QFunction,Policy):
    name="TRPOPolicy"
    def __init__(self,env):
        super(TRPOPolicy,self).__init__(env)
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(),lr=1e-5,alpha=0.99,weight_decay=1e-4)

class VFunction(B.BaseNetwork):
    name="VFunction"
    def __init__(self,env):
        super(VFunction, self).__init__(env.observation_space.shape,1, owner_name=self.name)
        
        self.conv = [nn.Sequential(nn.Conv2d(self.input_shape[0], 8, kernel_size=8, stride=4), nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=4, stride=2), nn.Tanh(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.Tanh())]
        x = U.out_features(self.conv[0],self.input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   B.Flatten(),
                                   nn.Linear(np.prod(x), 512),nn.Tanh(),
                                   nn.Linear(512,self.output_shape))

        self.compile()
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025,alpha=0.95,eps=0.01,momentum=0.95)


class QFunction_S(B.BaseNetwork):
    name="QFunction_S"
    
    def __init__(self, env):
        super(QFunction_S, self).__init__(env.observation_space.shape,env.action_space.n, owner_name=self.name)
        self.conv = [nn.Sequential(nn.Conv2d(self.input_shape[0], 8, kernel_size=8, stride=4), nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=4, stride=2), nn.Tanh(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.Tanh())]
        x = B.output_shape(self.conv,self.input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   B.Flatten(),
                                   nn.Linear(np.prod(x), 512),nn.Tanh(),
                                   nn.Linear(512,self.output_shape))

        self.compile()
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025,alpha=0.95,eps=0.01,momentum=0.95)

"""
class TRPONetS(DQNetS):
    name="TRPONet_S"

class TRPONetSValue(TRPONetS):
    name="TRPONet_S_Value"
    def __init__(self, input_shape, output_shape):
        super(TRPONetSValue, self).__init__(input_shape, 1)
"""