#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:16:58 2018

@author: thinkpad
"""
import sys
sys.path.append("../")

import torch
from base.functions import BaseDeep, Policy

from nn.nets import TRPONet, DQNet, TRPONetValue
from nn.nets import TRPONetS, DQNetS, TRPONetSValue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AtariPolicy(Policy):
    name ="TRPO_Policy"

    def __init__(self,env):        
        super(AtariPolicy,self).__init__(env)
        self.setup_model(TRPONet)
        self.compile()

class AtariValue(BaseDeep):
    name ="TRPO_Value"
    
    def __init__(self,env):        
        super(AtariValue,self).__init__(env)
        self.setup_model(TRPONetValue)
        self.compile()

class AtariQ(BaseDeep):
    name ="Atari Q"

    def __init__(self,env):        
        super(AtariQ,self).__init__(env)
        self.setup_model(DQNet)
        self.compile()




class AtariSPolicy(Policy):
    name ="TRPO_S_Policy"

    def __init__(self,env):        
        super(AtariSPolicy,self).__init__(env)
        self.setup_model(TRPONetS)
        self.compile()
        
class AtariSValue(BaseDeep):
    name ="TRPO_S_Value"
    
    def __init__(self,env):        
        super(AtariSValue,self).__init__(env)
        self.setup_model(TRPONetSValue)
        self.compile()

class AtariSQ(BaseDeep):
    name ="Atari_S_Q"

    def __init__(self,env):        
        super(AtariSQ,self).__init__(env)
        self.setup_model(DQNetS)
        self.compile()