
from math import ceil
from re import L
import torch
from torch import nn
import pickle
######################################################
###################### data  #########################
######################################################
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

######################################################
###################### math ##########################
######################################################
import itertools as it
from math import factorial

from modulefinder import Module
import torch
from torch import nn
from torch.nn import Module
import pandas as pd
import numpy as np
from torch.nn.utils.parametrizations import orthogonal
import os

import torch
from torch import clamp
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from modulefinder import Module
import torch
from torch import nn
from torch.nn import Module
import pandas as pd
import numpy as np
from torch.nn.utils.parametrizations import orthogonal
import os

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
# load pickle module
import pickle


class DatasetGen(Dataset):
    def __init__(self, size, d, n, cloud_1, cloud_2, ce=False, device = 'cpu') -> None:
        super(DatasetGen, self).__init__()
        self.ce = ce
        self.cloud_1 = cloud_1
        self.cloud_2 = cloud_2
        self.d = torch.tensor(d)
        self.n = torch.tensor(n)
        self.device = device
        self.size = torch.tensor(size).to(device=self.device)
        self.dataset = torch.zeros(self.size*2, self.d, self.n).to(device=self.device)
        self.label_set = torch.zeros(self.size*2, 2).to(device=self.device)
        self.labels()
        self.populate()
        #self.centralize()
        self.addOrthogonal() #good
        self.addGaussian(0, 0.001)
        self.addPerm() #good

        #self.to(self.device)
    def labels(self) -> None:
        for idx in range(self.size):
            self.label_set[idx,:] = torch.tensor([1,0], dtype=torch.float32).to(device=self.device)
            self.label_set[self.size + idx,:] = torch.tensor([0,1], dtype=torch.float32).to(device=self.device)     
    ###############################################################
    ##############initialize dataset ##############################
    ###############################################################
    def populate(self) -> None:
        for idx in range(self.size):
            self.dataset[idx,:,:] = torch.clone(self.cloud_1).to(device=self.device)
            self.dataset[self.size + idx,:,:] = torch.clone(self.cloud_2).to(device=self.device)
    ###############################################################
    #################centralize####################################
    ###############################################################
    def centralize(self) -> None:
        onesis = torch.ones(self.n, self.n, device = self.device)
        for idx in range(self.size):
            self.dataset[idx,:,:] -= torch.div(1, clamp(self.n, min=1))*torch.matmul(self.dataset[idx,:,:] , onesis).to(device=self.device)
            self.dataset[self.size + idx,:,:] -= torch.div(1, clamp(self.n,min=1))*torch.matmul(self.dataset[self.size + idx,:,:], onesis).to(device=self.device)      
            if idx == 999:
                print(self.dataset[idx,:,:])
                print(self.dataset[self.size + idx])
    ###############################################################
    ##############add Gaussian noise ##############################
    ###############################################################
    def addGaussian(self, mean, std) -> None:
        for idx in range(self.size):
           self.dataset[idx,:,:] += (torch.randn(self.d, self.n)*std + mean).to(device=self.device)
           self.dataset[self.size + idx,:,:] += (torch.randn(self.d, self.n)*std + mean).to(device=self.device)
    ###############################################################
    ##############add orthogonal transformation ###################
    ###############################################################
    def addOrthogonal(self) -> None :
        for idx in range(self.size):
           ortho = orthogonal(nn.Linear(self.d, self.d)).weight.to(device=self.device) #new mapping for each index
           self.dataset[idx,:,:] = torch.matmul(ortho, self.dataset[idx,:,:]).to(device=self.device)
           ortho = orthogonal(nn.Linear(self.d, self.d)).weight.to(device=self.device) #new mapping for each index           
           self.dataset[self.size + idx,:,:] = torch.matmul(ortho, self.dataset[self.size + idx,:,:]).to(device=self.device)
    ##################################################################
    #####################permute######################################
    ##################################################################
    def addPerm(self) -> None:
        for idx in range(self.size):
            index_list = torch.randperm(self.n).to(device=self.device)
            self.dataset[idx,:,:] =self.dataset[idx,:,index_list].clone().to(device=self.device)
            index_list = torch.randperm(self.n).to(device=self.device)
            self.dataset[self.size + idx,:,:] = self.dataset[self.size + idx,:,index_list] .clone().to(device=self.device)


    ###############################################################
    ####################Modify for dataloader #####################
    ###############################################################
    def __len__(self):
        return len(self.label_set)
    def __getitem__(self, idx):
        data = self.dataset[idx,:,:].clone().detach().to(device=self.device)
        labels = self.label_set[idx].clone().detach().to(device=self.device)
        data = torch.squeeze(data).to(device=self.device)
        return data, labels



#example of equal sets of sets of distances https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.166001

C = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1,1,0],
    [-1,-1,0],
    [1,2,0],
    [-1,2,0],
    [0,0,1]
])

D = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1,1,0],
    [-1,-1,0],
    [1,2,0],
    [-1,2,0],
    [0,0,-1]
])
#laplacians of 2-regular, 6 vertices non isomorphic graphs
D_1 = torch.tensor( [ [-2.,1.,0.,0.,0.,1.],\
                      [1.,-2.,1.,0.,0.,0.],\
                      [0.,1.,-2.,1.,0.,0.],\
                      [0.,0.,1.,-2.,1.,0.],\
                      [0.,0.,0.,1.,-2.,1.],\
                      [1.,0.,0.,0.,1.,-2.]  ])

D_2 = torch.tensor( [ [-2.,1.,1.,0.,0.,0.],\
                      [1.,-2.,1.,0.,0.,0.],\
                      [1.,1.,-2.,0.,0.,0.],\
                      [0.,0.,0.,-2.,1.,1.],\
                      [0.,0.,0.,1.,-2.,1.],\
                      [0.,0.,0.,1.,1.,-2.]  ])
#upper triangular cholesky decomposition of adjacency matrix
a = torch.linalg.cholesky(-D_1)
b = torch.linalg.cholesky(-D_2)


#laplacians of 3-regular, 6 vetrices non isomorphic graphs
T_1 = torch.tensor( [ [-3.,0.,0.,1.,1.,1.],\
                      [0.,-3.,0.,1.,1.,1.],\
                      [0.,0.,-3.,1.,1.,1.],\
                      [1.,1.,1.,-3.,0.,0.],\
                      [1.,1.,1.,0.,-3.,0.],\
                      [1.,1.,1.,0.,0.,-3.]  ])

T_2 = torch.tensor( [ [-3.,0.,1.,1.,1.,0.],\
                      [0.,-3.,0.,1.,1.,1.],\
                      [1.,0.,-3.,1.,0.,1.],\
                      [1.,1.,1.,-3.,0.,0.],\
                      [1.,1.,0.,0.,-3.,1.],\
                      [0.,1.,1.,0.,1.,-3.]  ])

c = torch.linalg.cholesky(-T_1)
d = torch.linalg.cholesky(-T_2)

#4-regular, 7-vertix graph
Y_1 = torch.tensor( [ [-4.,0.,0.,1.,1.,1.,1.],\
                      [0.,-4.,0.,1.,1.,1.,1.],\
                      [0.,0.,-4.,1.,1.,1.,1.],\
                      [1.,1.,1.,-4.,0.,1.,0.],\
                      [1.,1.,1.,0.,-4.,0.,1.],\
                      [1.,1.,1.,1.,0.,-4.,0.],\
                      [1.,1.,1.,0.,1.,0.,-4.]])

Y_2 = torch.tensor( [ [-4.,0.,1.,1.,1.,1.,0.],\
                      [0.,-4.,0.,1.,1.,1.,1.],\
                      [1.,0.,-4.,0.,1.,1.,1.],\
                      [1.,1.,0.,-4.,0.,1.,1.],\
                      [1.,1.,1.,0.,-4.,0.,1.],\
                      [1.,1.,1.,1.,0.,-4.,0.],\
                      [0.,1.,1.,1.,1.,0.,-4.]  ])

epsilon = torch.eye(7)*0.01


e = torch.linalg.cholesky(-Y_1 + epsilon)
f = torch.linalg.cholesky(-Y_2 + epsilon)

X_p = torch.tensor([[ 6.73555740e-17,  1.10000000e+00,  2.20000000e+00],
 [-2.02066722e-16, -1.10000000e+00, -2.20000000e+00],
 [ 2.03951216e+00,  6.27697301e+00,  0.00000000e+00],
 [-2.03951216e+00, -6.27697301e+00,  0.00000000e+00],
 [-3.30000000e+00,  4.04133444e-16,  4.40000000e+00],
 [ 3.30000000e+00, -8.08266887e-16, -4.40000000e+00]])


X_m = torch.tensor([[ 6.73555740e-17,  1.10000000e+00, -2.20000000e+00],
 [-2.02066722e-16, -1.10000000e+00,  2.20000000e+00],
 [ 2.03951216e+00,  6.27697301e+00,  0.00000000e+00],
 [-2.03951216e+00, -6.27697301e+00,  0.00000000e+00],
 [-3.30000000e+00,  4.04133444e-16,  4.40000000e+00],
 [ 3.30000000e+00, -8.08266887e-16, -4.40000000e+00]])



alpha = torch.tensor(

[[ 0.        ,  0.      ,    0.        ],
 [ 3.        ,  2.      ,   -4.        ],
 [ 0.        ,  2.      ,    5.        ],
 [-3.        ,  2.      ,   -4.        ],
 [ 4.778192  , -2.      ,   -1.4727123 ],
 [-2.9389262 , -2.      ,    4.045085  ],
 [-0.07591003, -2.      ,   -4.999424  ],
 [ 0.        ,  5      ,    0.        ]]


)


beta = torch.tensor(
    
[[ 0.         , 0.         , 0.        ],
 [ 3.         , 2.         ,-4.        ],
 [ 0.         , 2.         , 5.        ],
 [-3.         , 2.         ,-4.        ],
 [ 4.778192   ,-2.         ,-1.4727123 ],
 [-2.9389262  ,-2.         , 4.045085  ],
 [-0.07591003 ,-2.         ,-4.999424  ],
 [ 0.         ,-5         , 0.        ]]
)
# TODO look into the features in chirality 

body_4_nonchiral1 = torch.tensor(

[[ 0.        ,  0.      ,    0.        ],
 [ 3.        ,  2.      ,   -4.        ],
 [ 0.        ,  2.      ,    5.        ],
 [-3.        ,  2.      ,   -4.        ],
 [ 4.778192  , -2.      ,   -1.4727123 ],
 [-2.9389262 , -2.      ,    4.045085  ],
 [-0.07591003, -2.      ,   -4.999424  ],
 [ 0.        ,  5      ,    0.        ]]


)


body_4_nonchiral2 = torch.tensor(
    
[[ 0.         , 0.         , 0.        ],
 [ 3.         , 2.         ,-4.        ],
 [ 0.         , 2.         , 5.        ],
 [-3.         , 2.         ,-4.        ],
 [ 4.778192   ,-2.         ,-1.4727123 ],
 [-2.9389262  ,-2.         , 4.045085  ],
 [-0.07591003 ,-2.         ,-4.999424  ],
 [ 0.         ,-5         , 0.        ]]
)


body_4_chiral1 = torch.tensor(
[[ 0.,  0., 0.],
 [ 3.,  0., -4.],
 [ 0.,  0. ,5.],
 [-3.,  0. ,-4.],
 [ 0.,  5.,  0.]])

body_4_chiral2 = torch.tensor(
[[ 0. , 0.,  0.],
 [ 3. , 0., -4.],
 [ 0. , 0.,  5.],
 [-3. , 0., -4.],
 [ 0. , -5.,  0.]])


#@title Point Clouds

if torch.cuda.is_available():
    device= 'cuda'
else:
    device = 'cpu'

#first one from Pod. 2020 that egnn should separate
A = torch.tensor([[-2,0,-2],\
                  [2,0,2],\
                  [0,1,1],\
                  [1,1,0],\
                  [-1,-1,0] ], device=device, dtype=torch.float)

B = torch.tensor([[-2,0,-2],\
                  [2,0,2],\
                  [0,1,-1],\
                  [1,1,0],\
                  [-1,-1,0] ], device=device, dtype=torch.float)

# non integer valued
EE = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1.47633,1.84294,0],
    [-0.74309,1.455,0],
    [-0.72972,2.82605,0],
    [0,0,1]
], device=device, dtype=torch.float)

FF = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1.47633,1.84294,0],
    [-0.74309,1.455,0],
    [-0.72972,2.82605,0],
    [0,0,-1]
], device=device, dtype=torch.float)

#example of equal sets of sets of distances https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.166001

C = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1,1,0],
    [-1,-1,0],
    [1,2,0],
    [-1,2,0],
    [0,0,1]
])

D = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1,1,0],
    [-1,-1,0],
    [1,2,0],
    [-1,2,0],
    [0,0,-1]
])
#laplacians of 2-regular, 6 vertices non isomorphic graphs
D_1 = torch.tensor( [ [-2.,1.,0.,0.,0.,1.],\
                      [1.,-2.,1.,0.,0.,0.],\
                      [0.,1.,-2.,1.,0.,0.],\
                      [0.,0.,1.,-2.,1.,0.],\
                      [0.,0.,0.,1.,-2.,1.],\
                      [1.,0.,0.,0.,1.,-2.]  ])

D_2 = torch.tensor( [ [-2.,1.,1.,0.,0.,0.],\
                      [1.,-2.,1.,0.,0.,0.],\
                      [1.,1.,-2.,0.,0.,0.],\
                      [0.,0.,0.,-2.,1.,1.],\
                      [0.,0.,0.,1.,-2.,1.],\
                      [0.,0.,0.,1.,1.,-2.]  ])
#upper triangular cholesky decomposition of adjacency matrix
a = torch.linalg.cholesky(-D_1)
b = torch.linalg.cholesky(-D_2)

#laplacians of 2-regular, 8 vertices non isomorphic graphs
M_1 = torch.tensor(
[[-2,1,0,0,0,0,0,1],\
[1,-2,1,0,0,0,0,0],\
[0,1,-2,1,0,0,0,0],\
[0,0,1,-2,1,0,0,0],\
[0,0,0,1,-2,1,0,0],\
[0,0,0,0,1,-2,1,1],\
[0,0,0,0,0,1,-2,1],\
[1,0,0,0,0,0,1,-2]], dtype=torch.float
)


M_2 = torch.tensor(
[[-2,1,0,1,0,0,0,0],\
[1,-2,1,0,0,0,0,0],\
[0,1,-2,1,0,0,0,0],\
[1,0,1,-2,0,0,0,0],\
[0,0,0,0,-2,1,0,1],\
[0,0,0,0,1,-2,1,0],\
[0,0,0,0,0,1,-2,1],\
[0,0,0,0,1,0,1,-2]], dtype=torch.float
)
#upper triangular cholesky decomposition of adjacency matrix
gortler_4_a = torch.linalg.cholesky(-M_1)
gortler_4_b = torch.linalg.cholesky(-M_2)


X_p = torch.tensor([[ 6.73555740e-17,  1.10000000e+00,  2.20000000e+00],
 [-2.02066722e-16, -1.10000000e+00, -2.20000000e+00],
 [ 2.03951216e+00,  6.27697301e+00,  0.00000000e+00],
 [-2.03951216e+00, -6.27697301e+00,  0.00000000e+00],
 [-3.30000000e+00,  4.04133444e-16,  4.40000000e+00],
 [ 3.30000000e+00, -8.08266887e-16, -4.40000000e+00]])


X_m = torch.tensor([[ 6.73555740e-17,  1.10000000e+00, -2.20000000e+00],
 [-2.02066722e-16, -1.10000000e+00,  2.20000000e+00],
 [ 2.03951216e+00,  6.27697301e+00,  0.00000000e+00],
 [-2.03951216e+00, -6.27697301e+00,  0.00000000e+00],
 [-3.30000000e+00,  4.04133444e-16,  4.40000000e+00],
 [ 3.30000000e+00, -8.08266887e-16, -4.40000000e+00]])



alpha = torch.tensor(

[[ 0.        ,  0.      ,    0.        ],
 [ 3.        ,  2.      ,   -4.        ],
 [ 0.        ,  2.      ,    5.        ],
 [-3.        ,  2.      ,   -4.        ],
 [ 4.778192  , -2.      ,   -1.4727123 ],
 [-2.9389262 , -2.      ,    4.045085  ],
 [-0.07591003, -2.      ,   -4.999424  ],
 [ 0.        ,  5      ,    0.        ]]


)


beta = torch.tensor(
    
[[ 0.         , 0.         , 0.        ],
 [ 3.         , 2.         ,-4.        ],
 [ 0.         , 2.         , 5.        ],
 [-3.         , 2.         ,-4.        ],
 [ 4.778192   ,-2.         ,-1.4727123 ],
 [-2.9389262  ,-2.         , 4.045085  ],
 [-0.07591003 ,-2.         ,-4.999424  ],
 [ 0.         ,-5         , 0.        ]]
)
body_4_nonchiral1 = torch.tensor(

[[ 0.        ,  0.      ,    0.        ],
 [ 3.        ,  2.      ,   -4.        ],
 [ 0.        ,  2.      ,    5.        ],
 [-3.        ,  2.      ,   -4.        ],
 [ 4.778192  , -2.      ,   -1.4727123 ],
 [-2.9389262 , -2.      ,    4.045085  ],
 [-0.07591003, -2.      ,   -4.999424  ],
 [ 0.        ,  5      ,    0.        ]]


)


body_4_nonchiral2 = torch.tensor(
    
[[ 0.         , 0.         , 0.        ],
 [ 3.         , 2.         ,-4.        ],
 [ 0.         , 2.         , 5.        ],
 [-3.         , 2.         ,-4.        ],
 [ 4.778192   ,-2.         ,-1.4727123 ],
 [-2.9389262  ,-2.         , 4.045085  ],
 [-0.07591003 ,-2.         ,-4.999424  ],
 [ 0.         ,-5         , 0.        ]]
)


body_4_chiral1 = torch.tensor(
[[ 0.,  0., 0.],
 [ 3.,  0., -4.],
 [ 0.,  0. ,5.],
 [-3.,  0. ,-4.],
 [ 0.,  5.,  0.]])

body_4_chiral2 = torch.tensor(
[[ 0. , 0.,  0.],
 [ 3. , 0., -4.],
 [ 0. , 0.,  5.],
 [-3. , 0., -4.],
 [ 0. , -5.,  0.]])
# Incompleteness 2020 https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.125.166001/si.pdf


#@title EGNN Aleph code
#from models.gcl import E_GCL, unsorted_segment_sum
import torch
from torch import nn

import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F
# This version supports varying-size-sets as inputs
def tr12(input):
   return torch.transpose(input, 1, 2)


class embed_vec_sort_basic(nn.Module):
   # Calculates a permutation-invariant embedding of n vectors in R^d, using Nadav
   # and Ravina's sort embedding.
   # Input size: b x d x n, with b = batch size, d = input feature dimension, n = set size
   # Output size: b x d_out. Default d_out is 2*d*n+1
    def __init__(self, d, n, d_out = None):
        super().__init__()
      
        if d_out is None:
            d_out = 2*d*n+1
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.d = d
        self.n = n
        self.d_out = d_out
        torch.manual_seed(42)#agnostic to re-instantiation
        self.A = torch.randn([d, d_out], device=self.device)
        torch.manual_seed(42)
        self.w = torch.randn([1, n, d_out], device=self.device)
        if d>n:
            torch.manual_seed(42)
            self.w = torch.randn([1, d, d_out], device=self.device)


    def forward(self, input):
        if self.d > self.n: #if small cloud
            #pad input
            zeros = torch.zeros(input.size(0), self.d, self.d, device=self.device)
            zeros[:,:,:self.n] = input
            input = zeros

        prod = tr12( torch.tensordot( tr12(input), self.A, [[2], [0]] ) ) 
        [prod_sort, inds_sort] = torch.sort(prod, dim=2)
        out = torch.sum( prod_sort * tr12(self.w), dim=2)

        return out
    

class embed_vec_sort(nn.Module):
   # Calculates a permutation-invariant embedding of a set of vectors in R^d, using Nadav
   # and Ravina's sort embedding. It has two modes of operation:
   #
   # Constant set size: (varying_set_size = False)
   # Input size: b x d x n, with b = batch size, d = input feature dimension, n = set size
   # Output size: b x d_out. Default d_out is 2*d*n+1
   #
   # Varying set size: (varying_set_size = True)
   # Here the input can contain any number of vectors. 
   # n-1 is taken to be the number of linear regions in the weight function, and should be
   # larger or equal to the maximal input set size. 
   # Input size: b x (d+1) x n, with b = batch size, d = input feature dimension
   # Output size: b x d_out. Default d_out is 2*d*n+1
   # The first row of the input should contain 1 or 0, corresponding to existing or absent input vectors.
   #
   # The weight interpolation was taken as in the paper:
   # Learning Set Representations with Featurewise Sort Pooling, by Zhang et al., 2020   
   #
   # learnable_weights - Treats the weights as learnable parameters
   def __init__(self, d, n, d_out = None, varying_set_size = False, learnable_weights = False):
      super().__init__()
      if torch.cuda.is_available():
            self.device = torch.device('cuda')
      else:
            self.device = torch.device('cpu')
      if d_out is None:
         d_out = 2*d*n+1

      self.d = d
      self.n = n
      self.d_out = d_out
      self.varying_set_size = varying_set_size

      if learnable_weights:
         torch.manual_seed(42)
         self.A = nn.Parameter(torch.randn([d, d_out], requires_grad=True, device=self.device)) # TODO test learnability
         torch.manual_seed(42)
         self.w = nn.Parameter(torch.randn([1, n, d_out], requires_grad=True, device=self.device))
      else:
         torch.manual_seed(42)
         self.A = torch.randn([d, d_out], requires_grad=False, device=self.device)
         torch.manual_seed(42)
         self.w = torch.randn([1, n, d_out], requires_grad=False, device=self.device)
         self.register_buffer(name="sort-embedding matrix A", tensor=self.A)
         self.register_buffer(name="sort-embedding vector w", tensor=self.w)

   # Input: A vector k of size b, containing the set size at each batch
   # Output: A weight tensor w_out of size b x n x d_out, which is an interpolated
   #         version of w, such that w_out[r, :, t] corresponds to the weight vector self.w[0, :, t]
   #         after interpolation to the set size k[r].
   def interpolate_weights(self, k):
      b = torch.numel(k)

      # i corresponds to the sorted sample number in the output weight w_out
      i = (torch.arange(1, 1+self.n, device=self.device)).unsqueeze(0).unsqueeze(2)

      # j corresponds to the sorted sample number in the input weight self.w
      j = (torch.arange(1, 1+self.n, device=self.device)).unsqueeze(0).unsqueeze(0)

      # k contains the set size of each batch
      # Note that the roles of k and n here are replaced compared to those in the paper.
      k = k.unsqueeze(1).unsqueeze(1)
      
      interp_matrix = torch.clamp(1-torch.abs( (i-1)*(self.n-1)/(k-1) - (j-1) ), min=0)     
      w_out = torch.bmm(interp_matrix, self.w.repeat(b,1,1))

      return w_out

   def forward(self, input): #TODO add exponential
      if self.varying_set_size:
         assert input.shape[1] == self.d+1, 'Dimension 1 of input should be d+1'
         q = input[:,0,:]
         q = q.unsqueeze(1)
         X = input[:,1:,:]
      else:
         assert input.shape[1] == self.d, 'Dimension 1 of input should be d'
         X = input

      assert input.shape[2] == self.n, 'Dimension 2 of input should be n'

      prod = tr12( torch.tensordot( tr12(X), self.A, [[2], [0]] ) ) 

      if self.varying_set_size:
         prod[q.repeat(1,self.d_out,1) == 0] = np.inf
         k = torch.sum(q, dim=2).flatten()
         w = self.interpolate_weights(k)
      else:
         w = self.w   
      
      [prod_sort, inds_sort] = torch.sort(prod, dim=2)
      prod_sort[prod_sort == np.inf] = 0

      out = torch.sum( prod_sort * tr12(w), dim=2)

      return out

    

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, recurrent=False, coords_weight=1., coords_agg='mean', tanh=False, universal_node_num=25, delta=1., exp=False, device='cpu'):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.hidden_nf = hidden_nf
        self.epsilon = 1e-8
        self.delta = delta
        edge_coords_nf = 9 + self.hidden_nf
        self.exp = exp
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        torch.manual_seed(42)
        self.edge_mlp = nn.Sequential( #TODO batch normalization
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf), #needs to change given features
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        torch.manual_seed(42)
        self.node_mlp = nn.Sequential(#TODO batch normalization
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        torch.manual_seed(42)
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.manual_seed(42)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        torch.manual_seed(42)
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
        self.to(device)

    def edge_model(self, source, target, radial, embedding,edge_attr):
        ###radial === gram_dist
        if edge_attr is None:  # Unused.
            out = torch.cat([source.to(self.device), target.to(self.device), radial.to(self.device), embedding.to(self.device)], dim=1).to(self.device)
        else:
            out = torch.cat([source.to(self.device), target.to(self.device), radial.to(self.device), embedding.to(self.device),edge_attr.to(self.device)], dim=1).to(self.device)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val 
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        #agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0)) # summing
        agg = unsorted_segment_embed(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff
    def coord2gram(self, edge_index, coord):
        """"
        edge index: tuple of vectors specifying which edge indices are "linked"
        coord: concatenated point clouds in bach mode (contiguous)

        return: local frame per pair, shape (edge_index.size(0), d^2 + 2nd + 1)
        """
        row, col = edge_index
        #perp for global features
        final_index_list = count_num_vecs_cloud(edge_index, coord)
        vec_for_embed = prep_vec_for_embed(coord, final_index_list, edge_index)
        ug_vec = calc_ug(vec_for_embed, edge_index)
        gram, projections = torch.split(ug_vec, 3, dim=2) #shape b x d  x (n+1) TODO change to one identical to qm9
        #apply exponential
        if self.exp:
            projections = torch.exp(torch.div((-1)*projections, self.delta))

        #apply embedding to projections
        d_temp, n_temp = projections.size(-1), projections.size(-2)
        embed = embed_vec_sort(d_temp, n_temp, 2*d_temp*n_temp +1 )

        embedding= embed(projections)
        return torch.flatten(gram, start_dim=1), embedding #flatten


    def forward(self, h, edge_index, coord, edge_attr, node_attr=None): #coord=x
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        gram_dist, embedding = self.coord2gram(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], gram_dist, embedding, edge_attr) #pass coord here
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

def count_num_vecs_cloud(edge_index, coord):
    """"
    For each local frame, count and organize vectors in cloud not in local frame
    return index list of indices of batched point cloud for embedding
    """
    row, col = edge_index
    
    bincount = torch.bincount(row)#each index is number of times this int is present in row
    #bincount_col = torch.bincount(col) # should be identical
    #local frame index list
    init_index_list = torch.cat([row.unsqueeze(-1), col.unsqueeze(-1)], dim=1)
    #resize to fit n-1 incdices to prep for embedding
    empty = init_index_list.new_full(torch.Size([init_index_list.size(0), init_index_list.size(1) + bincount[0] -1 ]), 0) 
    #create tensor with lists of cloud indices to take to embed
    col = col.reshape(-1,bincount[0])
    cat_prep = col[row] #TODO contains superflous projections on tensor that is in gram matrix
    final_index_list = torch.cat([init_index_list, cat_prep], dim=1) #checked
    return final_index_list

def prep_vec_for_embed(coord, final_index_list, edge_index):
    """"
    Returns big vector of candidate for projections along batches
    """
    row, col = edge_index
    #populate big vector with indexed point cloud
    populate = coord[final_index_list]
    #add the perp vector
    perp_vec = torch.linalg.cross( coord[row] , coord[col] ).unsqueeze(-2)
    cat = torch.cat([ perp_vec,populate], dim=1) #checked
    return cat

def calc_ug(vec_for_embed, edge_index):
    """"
    returns vector with upper gram matrix of distances+norms, with superfluous projection
    """
    cat = vec_for_embed[:,:3,:]
    cdist = torch.cdist(cat, vec_for_embed, p=2)
    norms = torch.linalg.norm(cat, ord=None, axis=2, keepdims=False) #unchecked
    norms_diag = torch.diag_embed(norms)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    zeros = torch.zeros(cdist.size(), device=device)
    zeros[:,:,:3] = norms_diag
    ug_gram_dist = zeros.to(device) + cdist.to(device) # this is final dist gram matrix with dists
    return ug_gram_dist

    
class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, delta=1., exp=False, normalize=False, device='cpu'):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention, delta=delta, exp=exp, device=device)

        del self.coord_mlp
        torch.manual_seed(42)
        self.node_mlp = nn.Sequential(
            nn.Linear(nodes_attr_dim + hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        self.normalize = normalize
        self.act_fn = act_fn
        self.to(device)
        

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord
    def coord2gram(self, edge_index, coord): # needs testing!
        """"
        edge index: tuple of vectors specifying which edge indices are "linked"
        coord: concatenated point clouds in bach mode (contiguous)

        return: local frame per pair, shape (edge_index.size(0), d^2 + 2nd + 1)
        """
        row, col = edge_index
        #prep for global features
        final_index_list = count_num_vecs_cloud(edge_index, coord)
        vec_for_embed = prep_vec_for_embed(coord, final_index_list, edge_index)
        ug_vec = calc_ug(vec_for_embed, edge_index)
        (gram, projections) = torch.split(ug_vec, [3, ug_vec.size(2)-3], dim=2) #shape b x d  x (n+1)
        if self.exp:    
            projections = torch.exp(torch.div((-1)*projections, self.delta))
        if self.normalize:
            projections = F.normalize(projections)
        d_temp, n_temp = projections.size(-2), projections.size(-1)
        #begin with testing all ones
        #ones = torch.ones( projections.size(0), 1,projections.size(2) , device=self.device)
        non_empty_mask = projections.abs().sum(dim=1).bool().int().reshape(projections.size(0),1, n_temp) # list of 0/1 for when plgging into Tal's embedding

        fit = torch.cat([non_empty_mask, projections], dim=1)
        
        embed = embed_vec_sort(d_temp, n_temp, self.hidden_nf, varying_set_size=True, learnable_weights=True) #hidden_nf is the # of projections
        embedding= embed(fit)

        return torch.flatten(gram, start_dim=1), embedding #flatten

    def forward(self, h, edge_index, coord, node_mask=None, edge_mask=None, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        #radial, coord_diff = self.coord2radial(edge_index, coord)
        gram_dist, embedding = self.coord2gram(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], gram_dist, embedding,edge_attr) #pass coord here
        if edge_mask is not None:
            edge_feat = edge_feat * edge_mask #REMOVE?

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1, exp=False, delta=1.,h0=None, edges=None, edge_attr=None, node_mask=None, edge_mask=None, n_nodes=None):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.exp = exp
        self.n_layers = n_layers
        self.h0 = h0
        self.edges = edges
        self.edge_attr=edge_attr
        self.node_mask=node_mask
        self.edge_mask=edge_mask
        self.n_nodes=n_nodes
        ### Encoder
        torch.manual_seed(42)
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention, exp=self.exp, delta=delta, device=device))
        torch.manual_seed(42)
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        torch.manual_seed(42)
        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)
    
    def forward(self, x):
        h = self.embedding(self.h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, self.edges, x, self.node_mask, self.edge_mask, edge_attr=self.edge_attr, node_attr=self.h0, n_nodes=self.n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, self.edges, x, self.node_mask, self.edge_mask, edge_attr=self.edge_attr,
                                                      node_attr=None, n_nodes=self.n_nodes)

        h = self.node_dec(h)
        if self.node_mask is not None:
            h = h * self.node_mask
        h = h.view(-1, self.n_nodes, self.hidden_nf )
        #h = torch.sum(h, dim=1)
        h = h.transpose(dim0=-1, dim1=-2)
        embed = embed_vec_sort( self.hidden_nf,self.n_nodes, self.hidden_nf, varying_set_size=False, learnable_weights=True )
        h = embed(h)
        pred = self.graph_dec(h)
        return pred


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def unsorted_segment_embed(data, segment_ids, num_segments):
    bin_count = torch.bincount(segment_ids)# constant array
    assert(bin_count.size(0) == num_segments)
    data = data.reshape(bin_count.size(0),bin_count[0], data.size(1))
    data=data.transpose(dim0=-1,dim1=-2)
    embed = embed_vec_sort(d=data.size(1), n=data.size(2), d_out=data.size(1)) #data.size(2) === hidden_nf
    embedding = embed(data)
    return embedding

def get_edges_local_frame(n_nodes, x_dim, point_cloud: torch.Tensor): 
    '''''
    args:
    point_cloud: torch.Tensor, nxd
    n_nodes n
    x_dim d
    returns:
    tensor([tensor, tensor]) size: [2,n(n-1),x_dim,x_dim]
    '''''
    rows, cols = torch.empty(n_nodes * (n_nodes-1), x_dim), torch.empty(n_nodes * (n_nodes-1), x_dim)

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


def train(epoch, loader, model, loss_fun, lr_scheduler, optimizer, device='cuda:0', partition = 'train'):
    torch.manual_seed(42)

    res = {'loss': 0, 'counter': 0, 'loss_arr':[], 'accuracy': 0, 'acc_list' : []}
    batch_size = 1 
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train() 
            optimizer.zero_grad()

        else:
            model.eval()

        #change label to num
        label = data[1].to(device)
        label = torch.argmax(label.squeeze()).to(device, dtype=torch.float)
        #device = label.get_device()
        #print(str(device))
        data_now = data[0].to(device)
        data_now = data_now.reshape(data_now.size(0)*data_now.size(2), data_now.size(1))
        pred = model(data_now).squeeze().to(device)
        #pred = torch.clamp(pred, min=-5, max=6).to(device)
        classify =  1 if torch.linalg.norm(pred-1.) < torch.linalg.norm(pred-0.) else 0

        if partition == 'train':

            loss = loss_fun(pred.squeeze(), label.squeeze())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        else:
            loss = loss_fun(pred.squeeze(), label.squeeze())
        
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        #label = 1 if label[1] > label[0] else 0
        res['accuracy'] += (classify == label) # for batch size = 1
        res['acc_list'].append(torch.div(res['accuracy'], res['counter']))
        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition
        log_interval = 5
        if i % log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    return res['loss'] / res['counter'], res['accuracy'] / res['counter'], res['loss_arr'], res['acc_list']#loss,accuracy



def main(cloud1, cloud2, dimension,n):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.device(device)
        print("cuda enabled")
    else:
        device = torch.device('cpu')
    # Dummy parameters
    batch_size = 1
    n_nodes = n
    n_feat = 1
    x_dim = dimension
    hidden_nf = 32
    h = torch.zeros(batch_size *  n_nodes, n_feat, device=device)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)


    epochs = 5
    #test_interval = 10

    cloud_1 = torch.clone(cloud1).to(device, dtype=torch.float)
    cloud_2 = torch.clone(cloud2).to(device, dtype=torch.float)

    train_len = 5000
    test_len = 100
    dataset_train = DatasetGen( train_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce = True )
    dataset_test = DatasetGen( test_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce = True )
    dataset_eval = DatasetGen( test_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce= True )

    dataloader_train = DataLoader( dataset_train,  batch_size=batch_size, shuffle=True)
    dataloader_eval = DataLoader( dataset_eval,  batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader( dataset_test,  batch_size=batch_size, shuffle=True)

    dataloaders = { 'train' : dataloader_train, 'test' : dataloader_test, 'valid' : dataloader_eval}
    torch.manual_seed(42)
    output_size = hidden_nf
    softmax_dim = 2
    layer_sizes = torch.mul(torch.tensor([2, 1, 1]), output_size)
    model = nn.Sequential(
        EGNN(in_node_nf=n_feat, hidden_nf=hidden_nf,  in_edge_nf=1, n_layers=20, device=device,exp=True,h0=h, edges=edges, edge_attr=edge_attr, node_mask=None, edge_mask=None, n_nodes=n_nodes),
    ).to(device)
    lr = 2e-4
    wd = 1e-6
    #iterations = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    loss_fun = torch.nn.MSELoss()
    res = {'epochs': [], 'losess': [], 'accuracies': [],'best_val': 1e-10, 'best_test': 1e-10, 'best_epoch': 0}

    for epoch in range(0, epochs):
        _, _, loss_arr,accuracy_list = train(epoch, dataloaders['train'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'train')
        val_loss, val_acc, _, _ = train(epoch, dataloaders['valid'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'valid')
        test_loss, test_acc, _, _ = train(epoch, dataloaders['test'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'test')
        res['epochs'].append(epoch)
        res['accuracies'].append(test_acc)
        if val_acc > res['best_val']:
            res['best_val'] = val_acc
            res['best_test'] = test_acc
            res['best_epoch'] = epoch
        print("Val accuracy: %.4f \t test accuracy: %.4f \t epoch %d" % (val_acc, test_acc, epoch))
        print("Val accuracy: %.4f \t test accuracy: %.4f \t epoch %d" % (val_acc, test_acc, epoch))
        print("Best: val accuracy: %.4f \t test accuracy: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
    return res['best_test']

if __name__ == "__main__":
    res2 = {'EGNNaleph_no_norm_no_update' :   {'2_body_ex_1' : None, '2_body_ex_2' : None,'2_body_ex_3' : None , '2022' : None }}
    points = {'2_body_ex_1' : (A, B, 3,5), '2_body_ex_2' : (C, D, 3, 7),'2_body_ex_3' : (EE, FF, 3, 6) , '2022' : (X_p, X_m, 3, 6) }
    for key, value in res2.items():
        for key2, value2 in value.items():
            res2[key][key2] = main(points[key2][0], points[key2][1],points[key2][2],points[key2][3])
            f = open("/home/snirhordan/GramNN/gortler.pkl","wb")

            # write the python object (dict) to pickle file
            pickle.dump(res2,f)

            # close file
            f.close()
            print(res2)
