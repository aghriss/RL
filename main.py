#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

from envs.grid import GRID
from base.wrappers import EnvWrapper
from agents.ddqn import DDQN

from networks.nets import QFunction
#import gym

import gc
gc.enable()
gc.collect()
#game = "breakout"
#env = ALE(game,num_frames = 2, skip_frames = 4, render = False).
#env = ALE("seaquest.bin")


game = "grid"
env0 = EnvWrapper(GRID(grid_size= 36 ,max_time=5000,stochastic = False),size=36,mode="rgb", frame_count = 1,frame_skip=0)
#env0.reset();env0.render()
#env = gym.make("Breakout-v0")
#env = gym.make("Pong-v0")
#env.name = "Pong-v0"
#env.name = "Breakout-v0"
#env0 = AtariWrapper(env, frame_count = 4, crop= "Breakout-v0")
#env0 = AtariWrapper(env, frame_count = 4, crop= "Pong-v0",size=64)
agent = DDQN(env0, QFunction, 0.99, 32, update_double=2000, memory_min = 5000, memory_max=20000, train_steps= 10000000, eps_start = 1, eps_min=0.1, eps_decay = 2e-6)
#agent.load()
#agent = DQN(env, 0.99, 32, memory_max=100000, train_steps= 10000000, eps_start = 1, eps_decay = 1e-6)
#agent.load("WrapperBreakout-v0.pt")
#agent = TRPO(env,0.99,1024)

print(env0.observation_space)
#agent.load("GRID.pt")
agent.train()

