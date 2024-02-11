
# Based on https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, loss):

        score = -loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

import torch
from torch import nn

class Model(nn.Module):
  def __init__(self, X0, w, z, params):
    super().__init__()
    self.X = nn.Parameter(X0, requires_grad=True)
    self.params = params
    self.w = w
    self.z = z
    
  def forward(self):
    return self.X 

import numpy as np

def mollify(delta, x, d):
  return torch.exp(-torch.pow(x, 2)/(2*delta**2))/torch.pow(2*np.pi*(delta**2), torch.as_tensor(d/2))






# Kinetic energy (velocity^2)
def KE(traj, z_tensor, params):
  V = torch.diff(traj, dim=-1)
  l2_val = torch.sum(torch.pow(V, 2))
  l2_val += torch.sum(torch.pow(traj[:, :, 0] - z_tensor, 2))

  return l2_val/params['dt']/params['N']

# Nonlocal energy (continuum Gaussian)
def NLE_gauss(traj, w_tensor, params):
  dist_y = torch.cdist(traj[:, :, -1], traj[:, :, -1])
  reg_val = mollify(params['delta'], dist_y, params['d']).mean()

  dist_w = torch.cdist(traj[:, :, -1], w_tensor) 
  reg_val -= 2*mollify(params['sigma'], dist_w, params['d']).mean()
  
  return reg_val/params['eps']

# Nonlocal energy (sum of diracs)
def NLE(traj, w_tensor, params):
  d = w_tensor.shape[-1]

  dist_y = torch.cdist(traj[:, :, -1], traj[:, :, -1])
  reg_val = mollify(params['delta'], dist_y, d).mean()

  dist_w = torch.cdist(traj[:, :, -1], w_tensor) 
  reg_val -= 2*mollify(params['delta'], dist_w, d).mean()
  
  return reg_val/params['eps']

def NLE_pos(traj, w_tensor, params):
  return NLE(traj, w_tensor, params) + NLE_cons(w_tensor, params)






# Kinetic energy (acceleration^2)
def KE_acc(traj, z_tensor, params):
  X = torch.cat((z_tensor, traj), dim=-1) # add the first two time steps back
  V = torch.diff(X, dim=-1) # velocity
  acc = torch.diff(V, dim=-1) # accerlation
  l2_val = torch.sum(torch.pow(acc, 2))

  return l2_val/(params['dt']**3)/params['N']

# Nonlocal energy (for acceleration control)
def NLE_acc(traj, w_tensor, params):
  x1 = traj[:, :, -1]
  v1 = (traj[:, :, -1] - traj[:, :, -2])/params['dt']
  y = torch.cat((x1, v1), dim = -1)

  d = w_tensor.shape[-1]

  dist_y = torch.cdist(y, y)
  reg_val = mollify(params['delta'], dist_y, d).mean()

  dist_w = torch.cdist(y, w_tensor)
  reg_val -= 2*mollify(params['delta'], dist_w, d).mean()
  
  return reg_val/params['eps']

def NLE_acc_pos(traj, w_tensor, params):
  return NLE_acc(traj, w_tensor, params) + NLE_cons(w_tensor, params)






# Constant term of nonlocal energy
def NLE_cons(w_tensor, params):
  dist_w = torch.cdist(w_tensor, w_tensor)
  reg_val = mollify(params['delta'], dist_w, w_tensor.shape[-1]).mean()

  return reg_val/params['eps']






def obstacle(traj, center, r, params):
  o_val = r**2 - torch.sum(torch.pow(traj - center, 2), dim=1)
  o_val = torch.mean(torch.maximum(o_val, torch.zeros_like(o_val)))
  return o_val/params['eps_obst']





def endpoint_cost(y, w):
  N = y.shape[0]
  return np.sqrt(np.sum((y - w)**2)/N)

def allpoint_cost(X, w):
  z = X[:, :, 0]
  L = X.shape[-1]-1
  N = X.shape[0]
  w0 = draw_straight_lines(z, w, L+1)
  return np.sqrt(np.sum((X[:, :, 1:] - w0)**2)/L/N)




def draw_straight_lines(start, end, num):
  X0 = np.linspace(start, end, num=num, endpoint=True)
  X0 = X0.transpose((1, 2, 0))
  return X0[:,:,1:]
