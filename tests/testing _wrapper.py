#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:17:04 2018

@author: thinkpad
"""

from envs.grid import GRID
from envs.wrapper import EnvWrapper

env = GRID(grid_size=36, square_size=4, stochastic = False)

wrapper = EnvWrapper(env)

x = wrapper.reset()

y = wrapper.get_mouse()

