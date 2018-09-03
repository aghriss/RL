#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:54:14 2018

@author: thinkpad
"""
import sys
import torch
import torch.autograd
import numpy as np

sys.path.append("../")

from utils.console import summarize,Progbar
CHECK_PATH="./checkpoints/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BaseDeep(object):
    name = "Base"
    
    def __init__(self,env):
        
        self.input_shape = env.observation_space.shape
        self.output_shape = env.action_space.n
        
        self.progbar = Progbar(100)
        
    def setup_model(self, Net):
        self.net = Net(self.input_shape, self.output_shape).to(device)
        print(self.name, 'Net Set')

    def __call__(self,x):
        return self.net(x)
    
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
        
    def predict(self,image):
        if image.ndim == len(self.input_shape):
            image = image.reshape((1,)+image.shape)
            return self.net(torch.tensor(image, dtype=torch.float, device=device)).detach().cpu().numpy()
        else:
            return self.net(torch.tensor(image, dtype=torch.float, device=device)).detach().cpu().numpy()
    
    def optimize(self,loss,clip=False):
        self.optimizer.zero_grad()
        loss.backward()  
        if clip:torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

    
    def compile(self):
        self.net.compile()
        self.loss = self.net.loss
        self.optimizer = self.net.optimizer
        self.set_flattener()
        print("Compiled and Flattener Set")
        
    def save(self,name):
        print("Saving",CHECK_PATH+self.name+name+".pt")
        torch.save(self.net, CHECK_PATH+self.name+name+".pt")

    def load(self,name):
        print("Loading",CHECK_PATH+self.name+name+".pt")
        self.net = torch.load(CHECK_PATH+self.name+name+".pt").to(device)
        self.set_flattener()
        self.compile()
        
    def set_flattener(self):
        self.flattener = Flattener(self.net)
        
    def copy(self, model):
        self.net.load_state_dict(model.net.state_dict())
    
    def summary(self):
        print(self.net.name)
        summarize(self.net, self.input_shape)
        

class Policy(BaseDeep):

    def sample(self,state):
        logits = self.predict(state)
        soft = np.exp(logits)
        p = (soft/np.sum(soft))[0]
        
        
        #u = np.random.uniform(size=logits.shape)
        #return np.argmax(logits - np.log(-np.log(u)), axis=-1)
        #print(p)
        return np.random.choice(range(len(p)), p=p)
        
    def act(self,state):
        return np.argmax(self.predict(state), axis=-1)[0]

    def kl_logits(self, pi, states):
        logits1 = self.net(states)
        logits2 = pi(states)
        a0 = logits1 - logits1.max(axis=-1, keepdim=True)
        a1 = logits2 - logits2.max(axis=-1, keepdim=True)
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = ea0.sum(axis=-1, keepdim=True)
        z1 = ea1.sum(axis=-1, keepdims=True)
        p0 = ea0 / z0
        return (p0 * (a0 - torch.log(z0) - a1 + torch.log(z1))).sum(axis=-1)


class Flattener(object):
    
    def __init__(self, model):
        self.variables = model.parameters
        self.total_size = self.get().shape[0]
        self.idx = [0]
        self.shapes=[]
        for v in self.variables():
            self.shapes.append(v.shape)
            self.idx.append(self.idx[-1]+np.prod(self.shapes[-1]))

    def set(self,theta):
        
        assert theta.shape == (self.total_size,)
        
        for i,v in enumerate(self.variables()):
            v.data = torch.tensor(theta[self.idx[i]:self.idx[i+1]].view(self.shapes[i])).float().detach()
    
    def get(self):
        return flatten(self.variables())
    
    def flatgrad(self,f,retain=False,create=False):
        return flatten(torch.autograd.grad(f, self.variables(),retain_graph=retain,create_graph=create))
    
def flatten(x):
    return torch.cat([w.contiguous().view(-1) for w in x])
