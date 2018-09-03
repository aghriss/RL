#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:27:48 2018

@author: thinkpad
"""
import numpy as np
import collections
import torch

class BaseAgent(object):
    name ="BaseAgent"
    def __init__(self):
        
        self.history = collections.OrderedDict()
        self.functions = []
    def act(self,state,train=False):
        
        raise NotImplementedError
    
    def train(self,episodes):
        
        raise NotImplementedError
        
    def save(self):
        for f in self.functions:
            f.save(self.env.name+"."+self.name)
        
    def load(self):
        for f in self.functions:
            f.load(self.env.name+"."+self.name)

    def log(self, key,value=None):
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().numpy())
        if key not in self.history.keys():    
            self.history[key] = [value]
            #self.history = collections.OrderedDict(sorted(self.history.items()))
        else:
            self.history[key] = np.concatenate([self.history[key],[value]])

    def print(self):
        max_l = max(list(map(len,self.history.keys())))
        for k,v in self.history.items():
            print(k+(max_l-len(k))*" ",end="")
            if v is not None:
                print(" : %s"%str(v[-1]))
        
    def play(self,name='play'):        
        name = str(name)+self.env.name+self.name
        state = self.env.reset(record=True)
        done = False
        while not done:            
            action = self.act(state,train=False)
            state, _, done, info = self.env.step(action)
        self.env.save_episode(name)