#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:38:25 2018

@author: thinkpad
"""

from envs.grid import GRID
from envs.torchwrapper import AtariWrapper, EnvWrapper
from agents.trpo import TRPO
from models.atari import AtariQ, AtariValue, AtariPolicy
from models.atari import AtariSQ, AtariSValue, AtariSPolicy
import gc
from base.agent import Agent
from utils.console import Progbar
import utils.math as m_utils
import gym
gc.enable()
gc.collect()
game = "grid"



#env = gym.make("Breakout-v0")
#env = gym.make("Pong-v0")
#env.name = "Pong-v0"
#env.name = "Breakout-v0"
#env0 = AtariWrapper(env, size=56, frame_count = 4, crop= "Breakout-v0")
#env0 = AtariWrapper(env, size=64, frame_count = 3, crop= "Pong-v0")
env0 = EnvWrapper(GRID(grid_size=16,max_time=500,stochastic = False, square_size=3),size=48,mode="rgb", frame_count = 1)
agent = TRPO(env0, AtariPolicy, AtariValue,timesteps_per_batch=396, # what to train on
        gamma=0.99, lam=0.98, # advantage estimation
        max_kl=1e-4,
        cg_iters=10,
        cg_damping=0.01,
        vf_iters=1,
        checkpoint_freq=10)
#agent.load()
#agent.save("TRPO_GRID")

agent.train()

agent.play()
