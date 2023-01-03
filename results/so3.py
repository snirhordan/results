import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F


def tr12(input):
    return torch.transpose(input, 1, 2)

def gen_cross_product(input):
  """
  Generalized cross product
  Input 1 x d x n
  returns cross product of columns
  """
  if torch.cuda.is_available():
    device='cuda'
  else:
    device='cpu'
  mat = input.squeeze() # n-1 x n
  ones = torch.ones(1, mat.size(1), device=device)
  mat = torch.cat([ones, mat], dim=0)
  n=mat.size(1)
  order = torch.arange(n,device=device)
  empty= torch.zeros(n,device=device)
  first_row = temp =torch.cat([order[:0], order[0+1:]], dim=0)
  for i in range(n):
    temp = torch.cat([order[:i], order[i+1:]], dim=0)
    empty[i] =(-1)**i * torch.det(mat[first_row][:,temp])
  return empty
class embed_vec_sort(nn.Module):
   # Calculates a permutation-invariant embedding of n vectors in R^d, using Nadav
   # and Ravina's sort embedding.
   # Input size: b x d x n, with b = batch size, d = input feature dimension, n = set size
   # Output size: b x d_out. Default d_out is 2*d*n+1 
    def __init__(self, d, n, d_out = None):
        super().__init__()
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            self.device='cpu'
      
        if d_out is None:
            d_out = 2*d*n+1

        self.d = d
        self.n = n
        self.d_out = d_out

        self.A = torch.randn([d, d_out], device=self.device)
        self.w = torch.randn([1, n, d_out], device=self.device)

    def forward(self, input):
        prod = tr12( torch.tensordot( tr12(input), self.A, [[2], [0]] ) ) 
        [prod_sort, inds_sort] = torch.sort(prod, dim=2)
        out = torch.sum( prod_sort * tr12(self.w), dim=2)

        return out


# Calculates an embedding of n vectors in R^d that is invariant to permutations,
# rotations and optionally translations.
#
# Input size: b x (d + d_feature) x n
# b: batch size
# d: dimension of Euclidean space (currently only d=3 is supported)
# d_feature: accompanying feature dimension (default: 0)
# n: number of points
#
# Input shape: input[:,  0:(d-1), :] should be the Euclidean coordinates. The rest input[:,  d:(d+d_feature), :] should contain the accompanying feature vectors.
class embed_graph(nn.Module):
    def __init__(self, d, n, d_feature = 0, translation_invariant = True, is_compact = True):
        super().__init__()
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            self.device='cpu'
        self.d = d      
        self.n = n
        self.d_feature = d_feature

        self.translation_invariant = translation_invariant

        self.n_combs = np.prod(range(n-d+2,n+1))

        # Dimension of the embedding vector for one index-combination
        self.d_comb = d*d + (d-1)*d_feature + 2*(d+d_feature)*(n-(d-1)) + 1

        # Dimension of the embedding for the entire graph
        if is_compact:
            self.d_graph = 2*(d+d_feature)*n + 1
        else:
            self.d_graph = 2*self.d_comb*self.n_combs
        
        self.embed_comb = embed_vec_sort(d+d_feature, n-(d-1))
        self.embed_graph = embed_vec_sort(self.d_comb, self.n_combs, self.d_graph)


    def forward(self, input):
        input = input.reshape(1, input.size(dim=0), input.size(dim=1))
        [b,d_tot,n] = input.shape
        d_feature = self.d_feature
        d = d_tot - d_feature
        
        assert(d == self.d and n == self.n)
        #assert(d == 3) # Only d=3 is currently supported
        
        X = input[:,range(d),:]
        F = input[:,range(d,d_tot),:] if d_feature > 0 else None
        
        if self.translation_invariant:
            centers = torch.mean(X, axis=2)
            centers = torch.unsqueeze(centers, 2)
            X -= centers
        
        combs = torch.combinations(torch.arange(start=0, end=n), r=d-1, with_replacement=False)
        combs_flip = combs.flip(dims=[1])
        combs = torch.cat([combs, combs_flip], dim=0)

        global_vecs = torch.zeros([b, self.d_comb, self.n_combs], device=self.device)
        
        for i, comb in enumerate(combs):
            M0 = X[:, :, comb]

            #chnage to generalized cross 
            #xprod = torch.linalg.cross(M0[:,:,0], M0[:,:,1], dim=1)
            xprod = gen_cross_product(M0.squeeze().t()).t().reshape(1, self.d) #tested
            xprod = torch.unsqueeze(xprod, 2)
        
            # For a vector combination (v1, v2), the matrix M consists of columns [v1, v2, v1 x v2]
            M = torch.cat([M0, xprod], dim=2)
            Mt = tr12(M)
            MtM = torch.bmm(Mt,M)
            MtM_vec = torch.flatten(MtM, 1)
        
            if F is None:
                vec1 = MtM_vec
            else:
                vec1 = torch.cat([MtM_vec, torch.flatten(F[:,:,comb] , 1)], dim=1)
            
            # Complement of the current combination
            combcomp = [x for x in range(self.n) if x not in comb]
        
            # Product of M^T with all vectors except v1,v2
            MtV = torch.bmm(Mt, X[:,:,combcomp])
        
            if F is None:
                vecs_to_embed = MtV
            else:
                vecs_to_embed = torch.cat([MtV, F[:,:,combcomp]], dim=1)
        
            vec2 = self.embed_comb(vecs_to_embed)
        
            global_vecs[:, :, i] = torch.cat([vec1, vec2], dim=1)
        
        out = self.embed_graph(global_vecs)

        return out
    
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
#special orthogonal
import geotorch



class DatasetGen(Dataset):
    def __init__(self, size, d, n, cloud_1, cloud_2, ce=False, device = 'cpu', special=True) -> None:
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
        if special:
            self.addSpecialOrthogonal()
        else:
            self.addOrthogonal()
        self.addPerm()
        self.addGaussian(0, 0.001) #TODO Add Gaussian 
        self.centralize()

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
    #######################################################################
    ##############add special orthogonal transformation ###################
    #######################################################################
    def addSpecialOrthogonal(self) -> None :
        for idx in range(self.size):
           ortho = geotorch.SO(size=torch.empty(self.d,self.d).size()).sample().to(device=self.device)
           self.dataset[idx,:,:] = torch.matmul(ortho, self.dataset[idx,:,:]).to(device=self.device)
           ortho = geotorch.SO(size=torch.empty(self.d,self.d).size()).sample().to(device=self.device) #new mapping for each index           
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
        data_now = data[0].squeeze().to(device)
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

    epochs = 5
    #test_interval = 10

    cloud_1 = torch.clone(cloud1).to(device, dtype=torch.float)
    cloud_2 = torch.clone(cloud2).to(device, dtype=torch.float)

    train_len = 25000
    test_len = 100
    dataset_train = DatasetGen( train_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce = True )
    dataset_test = DatasetGen( test_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce = True )
    dataset_eval = DatasetGen( test_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce= True )

    dataloader_train = DataLoader( dataset_train,  batch_size=batch_size, shuffle=True)
    dataloader_eval = DataLoader( dataset_eval,  batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader( dataset_test,  batch_size=batch_size, shuffle=True)

    dataloaders = { 'train' : dataloader_train, 'test' : dataloader_test, 'valid' : dataloader_eval}
    output_size = 2*dimension*n + 1
    output_dim = 1
    layer_sizes = torch.mul(torch.tensor([2, 2, 1]), output_size)
    model = nn.Sequential(
        embed_graph(d=dimension, n=n),
        nn.Linear(output_size, layer_sizes[0]),
        nn.ReLU(),
        nn.Linear(layer_sizes[0], layer_sizes[1]),
        nn.ReLU(),
        nn.Linear(layer_sizes[1], output_dim)
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
    # TODO add gortler examples
    res2 = {'tal' :   {'gortler_3' : None, 'gortler_4': None}}
    points = {'gortler_3': (a, b, 6, 6), 'gortler_4': (gortler_4_a, gortler_4_b,8,8)}
    print("testing point clouds" + str(points))
    for key, value in res2.items():
        for key2, value2 in value.items():
            res2[key][key2] = main(points[key2][0], points[key2][1],points[key2][2],points[key2][3])
            f = open("/home/snirhordan/GramNN/tal.pkl","wb")

            # write the python object (dict) to pickle file
            pickle.dump(res2,f)

            # close file
            f.close()
            print(res2)
