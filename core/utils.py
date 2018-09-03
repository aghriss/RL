import sys
sys.path.append("../")

import torch
from torch import autograd
from tabulate import tabulate
import numpy as np

import collections
import core.console as S

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOUBLE = False
EPS = 1e-9
CHECK_PATH = "./checkpoints/"

if DOUBLE:
    if device.type=='cuda':
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

else:
    if device.type=='cuda':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
print(torch.get_default_dtype())

def bidirect_linesearch(func, x0, grad, expected,
                        alpha=0.5, beta_down=.5, beta_up=1.5, max_iter=50):
    f0 = func(x0)
    f1 = func(x0+grad)
    if f0-f1 > alpha*expected:
        beta = beta_up
        for m in range(1,max_iter):
            f1 = func(x0+grad*beta**m)
            if f0-f1 < alpha*expected*beta**m:
                return True, beta**(m-1)
    else:
        beta = beta_down
        # 1st step is good, perhaps we can jump higher
        for m in range(1,max_iter):
            f1 = func(x0+grad*beta**m)
            if f0-f1 > alpha*expected*beta**m:
                return True, beta**m
        return False, 0

def circlesearch(func, x0, grad1, grad2, expected, 
                 alpha = 0.5, beta=.5, max_iter=25):
    f0 = func(x0)
    t0 = torch.tensor(np.pi/2)
    for m in range(max_iter):
        f1 = func(x0+t0.cos()*grad1+t0.sin()*grad2)
        if f0-f1 > alpha*expected:
            return t0
        t0.mul_(beta)
    return torch.tensor(0.0)

def linesearch(func, x0, grad, expected, alpha = 0.5, beta=.5, max_iter=25):
    f0 = func(x0)
    coef = torch.tensor(1.0)
    for _ in range(max_iter):
        f1 = func(x0+grad*coef)
        if f0-f1 > alpha*expected*coef:# and f0-f1 < expected*coef:
            return coef
        coef.mul_(beta)
    return torch.tensor(0.0)

def get(x):
    y = x.detach().cpu().numpy()
    if y.ndim == 0:
        return y[()]
    return y

def get_grad(Func,Y):
    Z = autograd.Variable(Y,requires_grad=True)
    return autograd.grad(Func(Z),Z,retain_graph=False)[0]

def pinv(X,tol=1e-15):
    U,S,V = X.svd()
    S_inv = torch.zeros(S.shape)
    S_inv[S.abs()>tol] = 1/S[S.abs()>tol]
    return V.mm(S_inv.diag()).mm(U.t())

def print_loss(*args):
    print(tabulate([i for i in args], headers=['Loss', 'Old', 'New'], tablefmt='orgtbl'))

def queue(n):
    return collections.deque([0]*n, n)

def summary(model, input_size): S.summarize(model, input_size, device=device.type,double=DOUBLE)

def torchify(x, double=DOUBLE):
    if double:
        return torch.tensor(x,dtype=torch.float64)
    return torch.tensor(x).float()






