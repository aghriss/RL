# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:07:46 2018

@author: gamer
"""
import sys
sys.path.append("../")


import torch
import torch.nn as nn
import numpy as np

import core.utils as U
from core.console import Progbar

class BaseNetwork(nn.Module):
    """
        Base class for our Neural networks
    """
    name = "BaseNetwork"
    def __init__(self,input_shape, output_shape,owner_name=""):
        super(BaseNetwork,self).__init__()
         
        self.name = self.name+".%s"%owner_name
        self.input_shape = input_shape
        self.output_shape = output_shape       
        self.progbar = Progbar(100)


    def forward(self,x):
        return self.model(x)

    def optimize(self,l,clip=False):
        self.optimizer.zero_grad()
        l.backward()
        if clip:nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
    def fit(self,X,Y,batch_size=50,epochs=1,clip=False):
        Xtmp,Ytmp = X.split(batch_size),Y.split(batch_size)
        for _ in range(epochs):
            self.progbar.__init__(len(Xtmp))
            for x,y in zip(Xtmp,Ytmp):
                #self.optimizer.zero_grad()
                loss = self.loss(self.net(x),y)
                self.optimize(loss,clip)
                new_loss = self.loss(self.net(x),y)
                self.progbar.add(1,values=[("old",float(loss.detach().cpu().numpy())),("new",float(new_loss.detach().cpu().numpy()))])

    def step(self,grad):
        self.optimizer.zero_grad()
        self.flaten.set_grad(grad)
        self.optimizer.step()
    def set_learning_rate(self,rate, verbose=0):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
            if verbose : print("\n New Learning Rate ",param_group['lr'],"\n")
    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    def compile(self):
        self.loss = nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)
        self.flaten = Flattener(self.parameters)
        self.summary()
    def load(self,fname):
        print("Loading %s.%s"%(fname,self.name))
        dic = torch.load(fname+"."+self.name)
        super(BaseNetwork,self).load_state_dict(dic)
    def save(self,fname):
        print("Saving to %s.%s"%(fname,self.name))
        dic = super(BaseNetwork,self).state_dict()
        torch.save(dic, "%s.%s"%(fname,self.name))
    def copy(self,X):
        self.flaten.set(X.flaten.get())
    def summary(self):
        U.summary(self, self.input_shape) 

class Flattener(object):
    """
        Flattener Class
        
        handles operations related to gradient wrt the flattened parameters,
        getting and setting the network parameters
        
        Inputs:
            network.parameters
        Operations:
            Keeps the parameters with requires_grad = True
    """
    def __init__(self, parameters):
        self.variables = parameters_gen(parameters)
        self.total_size = self.get().shape[0]
        self.idx = [0]
        self.shapes=[]
        for v in self.variables():
            self.shapes.append(v.shape)
            self.idx.append(self.idx[-1]+np.prod(self.shapes[-1]))
    
    def set(self,theta):
        assert theta.shape == (self.total_size,)
        for i,v in enumerate(self.variables()):
            v.data = U.torchify(theta[self.idx[i]:self.idx[i+1]].view(self.shapes[i])).detach()
    
    def get(self):
        return flatten(self.variables())
    
    def flatgrad(self,f,retain=False,create=False):
        return flatten(torch.autograd.grad(f, self.variables(),retain_graph=retain,create_graph=create))
    
    def arrayflatgrad(self, f, symmetric=True):
        shape = f.shape+(self.total_size,)
        Res = U.torchify(np.zeros(shape))
        #assert shape[0]==shape[1]
        for i in range(shape[0]):
            for j in range(i,shape[0]):
                Res[i,j] = self.flatgrad(f[i,j], retain=True)
        return Res
    
    def set_grad(self,d_theta):
        assert d_theta.shape == (self.total_size,)
        for i,v in enumerate(self.variables()):
            v.grad = U.torchify(d_theta[self.idx[i]:self.idx[i+1]].view(self.shapes[i])).detach()

def conv3_2(in_planes, out_planes,bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2,
                     padding=1, bias=bias)
def deconv3_2(int_c, out_c):
    return nn.ConvTranspose2d(int_c,out_c,3,stride=2,padding=1,output_padding=1,bias=False)
def output_shape(net,input_shape):
    return net(torch.zeros((1,)+input_shape)).detach().shape[1:]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class ResNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(conv3_2(in_c, out_c),
                                   conv3_2(out_c, out_c))
        self.conv2 = nn.Conv2d(in_c, out_c, 4, stride=4)
    def forward(self, x):
        return self.conv1(x) + self.conv2(x)

def flatten(x):
    return torch.cat([w.contiguous().view(-1) for w in x])
def parameters_gen(parameters):
    def generator():
        for p in parameters():
            if p.requires_grad:
                yield p
    return generator
        

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


