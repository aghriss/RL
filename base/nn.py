# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:07:46 2018

@author: gamer
"""
import sys
sys.path.append("../")

import torch
import torch.nn as nn

# ================================================================
# Base class NN
# ================================================================
class BaseNN(nn.Module):
    
    def __init__(self,input_shape, output_shape):

        super(BaseNN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def conv_output_shape(self,X):
        return X(torch.randn((1,)+self.input_shape)).shape[1:]
    
    def linear_output_shape(self,X):
        #print(type(self.input_shape),self.input_shape)
        return X(torch.randn((1,)+self.input_shape)).clone().detach().view(-1).shape[0]
        

class Unfolder(nn.Module):
    
    def __init__(self,kernel_size):
        super(Unfolder, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(self.kernel_size, stride=self.kernel_size)
    def forward(self,x):
        return self.unfold(x)

def normc_initializer(var, std=1.0, axis=0):
    out = torch.randn(var.shape)
    var.data = std*out/torch.sqrt(torch.pow(out,2).sum(0, keepdim=True))
