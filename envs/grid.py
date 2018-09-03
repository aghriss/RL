#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:27:24 2018

@author: thinkpad
"""

import numpy as np
import sys
sys.path.append("../")

from base.spaces import Discrete, Continuous
ACTIONS = {0: (1,0),1:(-1,0),2:(0,1),3:(0,-1)}
class GRID(object):
    
    name = "GRID"
    
    def __init__(self, grid_size=36, max_time=2000, stochastic=True, square_size=2):
        self.max_time = max_time
        
        self.grid_size = grid_size
        self.stochastic = stochastic
        self.square = square_size
        self.action_space = Discrete(4)
        self.observation_space = Continuous((self.grid_size*self.square,self.grid_size*self.square,3))
        self.board = np.zeros((self.grid_size, self.grid_size))
        self.wall = np.zeros((self.grid_size, self.grid_size))
        self.state = np.zeros((self.grid_size*self.square,self.grid_size*self.square,3),dtype=np.int32)
        
        self.found = False

    def get_screen(self):

        self.state = self.state*0
        self.state[::self.square][:,::self.square][self.board>0,0] = 255
        self.state[::self.square][:,::self.square][self.board<0,2] = 255
        self.state[::self.square][:,::self.square][self.x, self.y] = 255

        for i in range(self.square-1):
            self.state[i+1::self.square] = self.state[::self.square]
            self.state[:,i+1::self.square] = self.state[:,::self.square]
        return self.state
    
    def step(self, action):
        
        """This function returns the new state, reward and decides if the
        game ends."""
        reward = 0
        if self.found:
            self.found = False
            self.reset_board()
        else:
            oldx,oldy = self.x,self.y
            if action in ACTIONS.keys():
                a = ACTIONS[action]
                self.x = self.x + a[0]
                self.y = self.y + a[1]
                reward = self.board[self.x,self.y]
                if reward == -1:
                    self.x,self.y = oldx, oldy
                    #reward = 0
            else:
                RuntimeError('Error: action not recognized')
    
            
        if reward == 1:
            self.found = True
        self.t = self.t + 1
        game_over = self.t > self.max_time
        
        return self.get_screen() , reward, game_over, self.found

    def reset(self):

        """This function resets the game and returns the initial state"""
        
        self.t = 0
        self.reset_board()
        return self.get_screen()
        
    def reset_board(self):
        self.board*=0

        step = self.grid_size//4
        self.wall[0,:] = self.wall[:,-1] = 1
        self.wall[2*step,:]=1
        self.wall[2*step,step:step+2]=0
        self.wall[2*step,3*step:3*step+2]=0
        self.wall = np.maximum(self.wall,self.wall.T)
        self.board[self.wall==1] = -1

        if self.stochastic:
            self.x,self.y = 0,0
            while self.board[self.x,self.y] < 0:
                self.x,self.y = np.random.randint(2, self.grid_size-3,2)
        else:
            self.x = 3
            self.y = 3

        self.add_mouse()
        

        
    def add_mouse(self):
        
        self.mouse_x,self.mouse_y = self.x,self.y
        OK = False
        if self.stochastic:
            while not OK: 
                self.mouse_x,self.mouse_y = np.random.randint(2, self.grid_size-3,2)
                OK = (self.mouse_x != self.x or self.mouse_y!=self.y) and self.board[self.mouse_x,self.mouse_y] > -1
        else:
            self.mouse_x = self.grid_size-3
            self.mouse_y = self.grid_size-3
            
        self.board[self.mouse_x,self.mouse_y] = 1
        self.found = False

    def get_mouse(self):
        return np.array([self.mouse_x,self.mouse_y])/self.grid_size

    def get_cat(self):
        return np.array([self.x,self.y])/self.grid_size

class GRID2(GRID):
    name = "GRID2"
    def __init__(self,*args,**kwargs):
        super(GRID2,self).__init__(*args,**kwargs)
        self.observation_space = Continuous((2,2,1))
    
    def current_state(self):
        
        return np.array([self.x, self.y, self.mouse_x, self.mouse_y]).reshape(2,2,1)
    
