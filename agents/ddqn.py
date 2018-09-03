#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:04:10 2018
,a
@author: thinkpad
"""
import sys
import collections
sys.path.append("../")

from base.baseagent import BaseAgent
from base.replaymemory import ReplayMemory
from core.console import Progbar
import core.utils as U
import numpy as np

class DDQN(BaseAgent):
    """
    Double Deep Q Networks
    """
    name = "DDQN"

    def __init__(self, env, deep_func, gamma, batch_size, memory_min, memory_max, update_double = 10000, train_steps=1000000, log_freq = 1000, eps_start = 1, eps_decay = -1, eps_min = 0.1):

        super(DDQN,self).__init__()

        self.env = env
        self.Q = self.model = deep_func(env)
        self.target_Q = deep_func(env)
        
        self.discount = gamma
        self.memory_min = memory_min        
        self.memory_max = memory_max
        self.eps = eps_start
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.done = 0
        self.log_freq = log_freq
        self.progbar = Progbar(self.memory_max)
        self.memory = ReplayMemory(self.memory_max,["state","action","reward","next_state","terminated"])

        self.eps_decay = eps_decay        
        if eps_decay == -1:            
            self.eps_decay = 1/train_steps
    
        self.eps_min = eps_min
        self.update_double = update_double
        self.actions=[]
        self.path_generator = self.roller()
        self.past_rewards = collections.deque([],50)
        self.functions = [self.Q]
    def act(self,state,train=True):
        
        if train:
            if np.random.rand()<self.eps:
                return np.random.randint(self.env.action_space.n)
        
        return np.argmax(self.Q.predict(state))
    
    def train(self):

        self.progbar.__init__(self.memory_min)
        while(self.memory.size < self.memory_min):
            self.path_generator.__next__()

        while(self.done<self.train_steps):

            to_log = 0
            self.progbar.__init__(self.update_double)
            old_theta = self.Q.flaten.get()
            self.target_Q.copy(self.Q)
            while to_log <self.update_double:

                self.path_generator.__next__()
                
                rollout = self.memory.sample(self.batch_size)
                state_batch = U.torchify(rollout["state"])
                action_batch = U.torchify(rollout["action"]).long()
                reward_batch = U.torchify(rollout["reward"])

                non_final_batch = U.torchify(1-rollout["terminated"])
                next_state_batch = U.torchify(rollout["next_state"])

                #current_q = self.Q(state_batch)
                
                current_q = self.Q(state_batch).gather(1, action_batch.unsqueeze(1)).view(-1)
                _, a_prime = self.Q(next_state_batch).max(1)
                
                
                # Compute the target of the current Q values
                next_max_q = self.target_Q(next_state_batch).gather(1,a_prime.unsqueeze(1)).view(-1)
                target_q =  reward_batch + self.discount * non_final_batch * next_max_q.squeeze()
                
                # Compute loss
                loss = self.Q.loss(current_q,target_q.detach()) # loss = self.Q.total_loss(current_q, target_q)
                
                # Optimize the model
                self.Q.optimize(loss,clip=True)
                
                self.progbar.add(self.batch_size,values=[("Loss",U.get(loss))])
                
                to_log+=self.batch_size

            self.target_Q.copy(self.Q)
            new_theta = self.Q.flaten.get()

            self.log("Delta Theta L1", U.get((new_theta-old_theta).abs().mean()))
            self.log("Av 50ep  rew",np.mean(self.past_rewards))
            self.log("Max 50ep rew",np.max(self.past_rewards))
            self.log("Min 50ep rew",np.min(self.past_rewards))
            self.log("Epsilon",self.eps)
            self.log("Done",self.done)
            self.log("Total",self.train_steps)
            self.target_Q.copy(self.Q)
            self.print()
            #self.play()
            self.save()
            
    def set_eps(self,x):
        self.eps = max(x,self.eps_min)
        
    def roller(self):
        
        state = self.env.reset()
        ep_reward = 0
        while True:
            episode = self.memory.empty_episode()
            for i in range(self.batch_size):
            
                # save current state
                episode["state"].append(state)
            
                # act
                action = self.act(state)   
                self.actions.append(action)
                state, rew, done, info = self.env.step(action)
    
                episode["next_state"].append(state)            
                episode["action"].append(action)
                episode["reward"].append(rew)        
                episode["terminated"].append(done)
                
                ep_reward+=rew
                self.set_eps(self.eps-self.eps_decay)
                
                if done:
                    self.past_rewards.append(ep_reward)
                    state = self.env.reset()
                    ep_reward = 0
                self.done += 1
                if not(self.done)%self.update_double:
                    self.update=True
            
            # record the episodes
            self.memory.record(episode)
            if self.memory.size< self.memory_min: self.progbar.add(self.batch_size,values=[("Loss",0.0)])        
            yield True
