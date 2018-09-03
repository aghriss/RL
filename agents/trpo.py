#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:52:25 2018

@author: thinkpad
"""
import sys,time
sys.path.append("../")
import numpy as np
import torch

import collections

from base.baseagent import BaseAgent
from core.console import Progbar
import core.math as m_utils
import core.utils as U

GAMMA=0.99
LAM=0.97
TIMESTEPS_BATCH=1000
MAX_KL=1e-2
CG_ITER=10
CG_DAMPING=1e-2
ENTROPY_COEF=0.0
VF_ITER=3

class TRPO(BaseAgent):

    name = "TRPO"

    def __init__(self, env, policy_func, value_func,
        timesteps_per_batch=1000, # what to train on
        gamma=0.99, lam=0.97, # advantage estimation
        max_kl=1e-2,
        cg_iters=10,
        entropy_coeff=0.0,
        cg_damping=1e-2,
        vf_iters=3,
        max_train=1000,
        checkpoint_freq=50):

        super(TRPO,self).__init__()
        
        self.env = env
        self.gamma = gamma
        self.lam = lam

        
        self.timesteps_per_batch = timesteps_per_batch
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.entropy_coeff = entropy_coeff
        self.cg_damping = cg_damping
        self.vf_iters = vf_iters
        self.max_train = max_train
        self.checkpoint_freq=checkpoint_freq
        
        self.policy = policy_func(env)
        self.oldpolicy = policy_func(env)
        self.value_function = value_func(self.env)
        self.progbar = Progbar(self.timesteps_per_batch)
        
        self.path_generator = self.roller()
        self.value_function.summary()
        
        self.episodes_reward=collections.deque([],50)
        self.episodes_len=collections.deque([],50)
        self.done = 0
        
        self.functions = [self.policy, self.value_function]
    def act(self,state,train=True):
        if train:
            return self.policy.sample(state)
        return self.policy.act(state)
        
    def calculate_losses(self, states, actions, advantages, tdlamret):

        pi = self.policy(states)
        old_pi = self.oldpolicy(states).detach()
        
        kl_old_new = m_utils.kl_logits(old_pi,pi)
        entropy = m_utils.entropy_logits(pi)
        mean_kl = kl_old_new.mean()
                
        mean_entropy = entropy.mean()
                
        ratio = torch.exp(m_utils.logp(pi,actions) - m_utils.logp(old_pi,actions)) # advantage * pnew / pold
        surrogate_gain = (ratio * advantages).mean()
            
        optimization_gain = surrogate_gain + self.entropy_coeff*mean_entropy
                
        losses = {"gain": optimization_gain, "meankl":mean_kl, "entropy": mean_entropy, "surrogate": surrogate_gain, "mean_entropy": mean_entropy}
        return losses

    def train(self):
        while self.done < self.max_train:
            print("="*40)
            print(" "*15, self.done,"\n")
            self._train()
            if not self.done%self.checkpoint_freq:
                self.save()
            self.done = self.done+1
        self.done = 0 
    def _train(self):

        # Prepare for rollouts
        # ----------------------------------------

        self.oldpolicy.copy(self.policy)
                
        path = self.path_generator.__next__()
                
        states = U.torchify(path["state"])
        actions = U.torchify(path["action"]).long()
        advantages = U.torchify(path["advantage"])
        tdlamret = U.torchify(path["tdlamret"])
        vpred = U.torchify(path["vf"]) # predicted value function before udpate
        advantages = (advantages - advantages.mean()) / advantages.std() # standardized advantage function estimate        
                        
        losses = self.calculate_losses(states, actions, advantages, tdlamret)       
        kl = losses["meankl"]
        optimization_gain = losses["gain"]

        loss_grad = self.policy.flaten.flatgrad(optimization_gain,retain=True)     
        grad_kl = self.policy.flaten.flatgrad(kl,create=True,retain=True)

        theta_before = self.policy.flaten.get()
        self.log("Init param sum", theta_before.sum())
        self.log("explained variance",(vpred-tdlamret).var()/tdlamret.var())
        
        if np.allclose(loss_grad.detach().cpu().numpy(), 0,atol=1e-15):
            print("Got zero gradient. not updating")
        else:
            print("Conjugate Gradient",end="")
            start = time.time()
            stepdir = m_utils.conjugate_gradient(self.Fvp(grad_kl), loss_grad, cg_iters = self.cg_iters)
            elapsed = time.time()-start
            print(", Done in %.3f"%elapsed)
            self.log("Conjugate Gradient in s",elapsed)
            assert stepdir.sum()!=float("Inf")
            shs = .5*stepdir.dot(self.Fvp(grad_kl)(stepdir))
            lm = torch.sqrt(shs / self.max_kl)
            self.log("lagrange multiplier:", lm)
            self.log("gnorm:", np.linalg.norm(loss_grad.cpu().detach().numpy()))
            fullstep = stepdir / lm
            expected_improve = loss_grad.dot(fullstep)
            surrogate_before = losses["surrogate"]
            stepsize = 1.0
            
            print("Line Search",end="")
            start = time.time()
            for _ in range(10):
                theta_new = theta_before + fullstep * stepsize
                self.policy.flaten.set(theta_new)
                losses = self.calculate_losses(states,actions,advantages, tdlamret)
                surr = losses["surrogate"] 
                improve = surr - surrogate_before
                kl = losses["meankl"]
                if surr == float("Inf") or kl ==float("Inf"):
                    print("Infinite value of losses")
                elif kl > self.max_kl:
                    print("Violated KL")
                elif improve < 0:
                    print("Surrogate didn't improve. shrinking step.")
                else:
                    print("Expected: %.3f Actual: %.3f"%(expected_improve, improve))
                    print("Stepsize OK!")
                    self.log("Line Search","OK")
                    break
                stepsize *= .5
            else:
                print("couldn't compute a good step")
                self.log("Line Search","NOPE")
                self.policy.flaten.set(theta_before)
            elapsed = time.time()-start
            print(", Done in %.3f"%elapsed)
            self.log("Line Search in s",elapsed)
            self.log("KL",kl)
            self.log("Surrogate",surr)
        start = time.time()
        print("Value Function Update",end="")
        self.value_function.fit(states[::5], tdlamret[::5], batch_size = 50, epochs = self.vf_iters)
        elapsed = time.time()-start
        print(", Done in %.3f"%elapsed)
        self.log("Value Function Fitting in s",elapsed)
        self.log("TDlamret mean",tdlamret.mean())
        self.log("Last 50 rolls mean rew",np.mean(self.episodes_reward))
        self.log("Last 50 rolls mean len",np.mean(self.episodes_len))
        self.print()

    def roller(self):
        
        state = self.env.reset()
        ep_rews = 0
        ep_len = 0
        while True:            
            
            path = {s : [] for s in ["state","action","reward","terminated","vf","next_vf"]}
            self.progbar.__init__(self.timesteps_per_batch)
            for _ in range(self.timesteps_per_batch):
                
                path["state"].append(state)
                # act
                action = self.act(state)
                vf = self.value_function.predict(state)
                state, rew, done,_ = self.env.step(action)
                path["action"].append(action)
                path["reward"].append(rew)
                path["vf"].append(vf)
                path["terminated"].append(done*1.0)
                path["next_vf"].append((1-done)*vf)

                ep_rews+=rew
                ep_len+=1
                                
                if done:
                    state = self.env.reset()
                    self.episodes_reward.append(ep_rews)
                    self.episodes_len.append(ep_len)
                    ep_rews = 0
                    ep_len = 0
                self.progbar.add(1)
                
            for k,v in path.items():
                path[k] = np.array(v)

            self.add_vtarg_and_adv(path)            
            yield path

    def Fvp(self,grad_kl):
        def fisher_product(v):
            kl_v = (grad_kl * v).sum()
            grad_grad_kl = self.policy.flaten.flatgrad(kl_v, retain=True)
            return grad_grad_kl + v * self.cg_damping
        
        return fisher_product

    def add_vtarg_and_adv(self, path):
        # General Advantage Estimation
        terminal = np.append(path["terminated"],0)
        vpred = np.append(path["vf"], path["next_vf"])
        T = len(path["reward"])
        path["advantage"] = np.empty(T, 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-terminal[t+1]
            delta = path["reward"][t] + self.gamma * vpred[t+1] * nonterminal - vpred[t]
            path["advantage"][t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        path["tdlamret"] = (path["advantage"] + path["vf"]).reshape(-1,1)
