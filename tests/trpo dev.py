#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:16:21 2018

@author: thinkpad
"""
from envs.grid import GRID
from envs.wrapper import EnvWrapper
from collections import deque
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

class test(object):
    pass

self=test()
from models.atari import AtariQ, AtariValue, AtariPolicy
value_func, policy_func = AtariValue, AtariPolicy

env = EnvWrapper(GRID(grid_size=64, square_size=4, stochastic = True), frame_count = 1)

timesteps_per_batch = 1000
max_kl, cg_iters = 1e-2, 10
gamma, lam = 0.99, 0.97
entropy_coeff=0.0,
cg_damping=1e-2,
vf_stepsize=3e-4,
vf_iters =3,
max_timesteps=0
max_episodes=0
max_iters=0

self.env = env
self.gamma = gamma

self.pi = policy_func(env)
self.oldpi = policy_func(env)
self.value_function = value_func(self.env)

self.timesteps_per_batch = timesteps_per_batch
self.max_kl = max_kl
self.cg_iters = cg_iters
self.entropy_coeff = entropy_coeff
self.cg_damping = cg_damping
self.vf_stepsize = vf_stepsize
self.vf_iters = vf_iters
self.max_timesteps = max_timesteps
self.max_episodes = max_episodes
self.max_iters = max_iters
self.lam = 0.98
def act(state):
    return self.pi.sample(state)
self.act = act

horizon = self.timesteps_per_batch
state = self.env.reset()
action = self.env.action_space.sample()
i = 0
path = {s : [] for s in ["prev_action", "state","action","reward","vf","terminated","next_vf","ep_rew","ep_len"]}
rews = 0
while True:
    path["prev_action"].append(action)
    path["state"].append(state)
    # act
    action = self.act(state)
    vf = self.value_function.predict(state)[0]
    
    state, rew, done, _ = self.env.step(action)
    rews += rew
    path["action"].append(action)
    path["reward"].append(rew)
    path["vf"].append(vf)
    path["terminated"].append(done*1.0)
    path["next_vf"].append((1-done)*vf)
                    
    if done:
        print("Done",i)
        path["ep_rew"].append(rews)
        path["ep_len"].append(i+1)
        state = self.env.reset()
    if not (i+1)%(horizon+1):
        # General Advantage Estimation
        print("horizon",i)
        for k,v in path.items():
                path[k] = np.array(v)
        terminal = np.append(path["terminated"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(path["vf"], path["next_vf"])
        T = path["reward"].shape[0]
        path["advantage"] = np.zeros(T)
        rew = path["reward"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-terminal[t+1]
            delta = rew[t] + self.gamma * vpred[t+1] * nonterminal - vpred[t]
            path["advantage"][t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        path["tdlamret"] = path["advantage"] + path["vf"]

        break

    i = i + 1
        

self.act = act
    
episodes_so_far = 0
timesteps_so_far = 0
iters_so_far = 0
self.tstart = time.time()
lenbuffer = deque(maxlen=40) 
rewbuffer = deque(maxlen=40) 
    

self.oldpi.copy(self.pi)
                
