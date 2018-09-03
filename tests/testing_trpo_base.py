#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:17:00 2018

@author: gamer
"""
import sys
sys.path.append("../")
import gc
gc.enable()
gc.collect()
from mpi4py import MPI
from baselines import logger
from baselines.trpo_mpi import trpo_mpi
from envs.grid import GRID
from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy
from envs.torchwrapper import AtariWrapper, EnvWrapper

game = "grid"
env = EnvWrapper(GRID(grid_size=16,max_time=1000,square_size=2,stochastic = False),size=32, mode="rgb",torch=False,frame_count=1,frame_skip=0)
#env = GRID(grid_size=36,max_time=1000,square_size=4,stochastic = True)

print(env.observation_space)

num_timesteps = 1e5
import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    logger.configure()
else:
    logger.configure(format_strs=[])
    logger.set_level(logger.DISABLED)
    
def policy_fn(name, ob_space, ac_space):
    return CnnPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space)
trpo_mpi.learn(env, policy_fn, timesteps_per_batch=512, max_kl=0.01, cg_iters=10, cg_damping=0.1,
    max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

