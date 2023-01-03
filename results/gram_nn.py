
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
        self.centralize()
        self.addOrthogonal()
        self.addGaussian(0, 0.1)
        self.addPerm()

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


#@title Point Clouds


#first one from Pod. 2020 that egnn should separate
A = torch.tensor([[-2,0,-2],\
                  [2,0,2],\
                  [0,1,1],\
                  [1,1,0],\
                  [-1,-1,0] ])

B = torch.tensor([[-2,0,-2],\
                  [2,0,2],\
                  [0,1,-1],\
                  [1,1,0],\
                  [-1,-1,0] ])

# non integer valued
E = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1.47633,1.84294,0],
    [-0.74309,1.455,0],
    [-0.72972,2.82605,0],
    [0,0,1]
])

F = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1.47633,1.84294,0],
    [-0.74309,1.455,0],
    [-0.72972,2.82605,0],
    [0,0,-1]
])

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

# Incompleteness 2020 https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.125.166001/si.pdf



torch.manual_seed(42)
class UpperGram(nn.Module): #tested
    def __init__(self, dimension, n, index_list) -> None:
        super(UpperGram, self).__init__()
        self.dim = dimension
        self.index_list = index_list
        self.n = n
    ###############################################################
    #########returns matrix of columns of idx list#################
    ###############################################################
    def chosen(self, index_list, cloud, a, b) -> torch.tensor:
        index_list = list(index_list)
        mat = torch.zeros(a, b)
        for idx in range(b):
            mat[:,idx] = cloud[:, index_list[idx]].squeeze()
        return mat
    ###############################################################
    #########returns Gram matrix by chodsen idx list ##############
    ###############################################################
    def gram(self, mat) -> torch.tensor:
        gram = self.chosen(self.index_list, mat,self.dim, self.dim)
        gram = torch.matmul(torch.t(gram), gram)
        return gram
    ###############################################################
    ####returns dx(nxd) projections matrix by chodsen idx list ####
    ###############################################################
    def projections(self, mat) -> torch.tensor:
        sorted_rest = [idx for idx in range(self.n) if idx not in self.index_list] #if repeating then too many indices
        sorted_rest = torch.tensor(sorted_rest)
        sorted_rest, indices = torch.sort(sorted_rest) #why sort the rest?
        matrix = self.chosen(sorted_rest, mat, self.dim, len(sorted_rest))
        matrix_d = self.chosen(self.index_list, mat, self.dim, self.dim)
        return torch.matmul(torch.t(matrix_d), matrix)
    ###############################################################
    #################returns Upper Gram matrix ####################
    ###############################################################
    def forward(self, mat) -> torch.tensor:
        gram = self.gram(mat)
        projections = self.projections(mat)
        return torch.cat((gram,projections), dim=1)

class Embed(nn.Module): #tested
    def __init__(self,dim, n, dim_orig = None, n_orig = None ,unique = False ) -> None:
        super(Embed, self).__init__()
        self.dim = dim
        self.n = n
        self.dim_orig = dim_orig
        self.n_orig = n_orig
        self.unique = unique
        self.original = (self.dim_orig != None and self.n_orig != None)
        # learned weights
        if not self.original:
          self.weights_sort = torch.randn(2*self.dim*self.n + 1 , self.dim, requires_grad=True)*0.01 #normal distribution, maybe uiform will work better
          self.weights_dot = torch.randn(2*self.dim*self.n + 1 , self.n, requires_grad=True)*0.01
        elif self.original:
          self.weights_sort = torch.randn(2*self.dim_orig*self.n_orig + 1 , self.dim, requires_grad=True)*0.01 #normal distribution, maybe uiform will work better
          self.weights_dot = torch.randn(2*self.dim_orig*self.n_orig + 1 , self.n, requires_grad=True)*0.01
        ##########################################################################
        #######################embed onto space###################################
        ##########################################################################
    def forward(self, mat : torch.tensor) -> torch.tensor:
        '''
        Embeds a matrix into 2nd+1 space, such that invariant to column permutations
        Recieves: matrix
        Returns: embedding
        '''
        assert(tuple(mat.size()) == tuple([self.dim, self.n]))
        
        if not self.unique and not self.original:
          self.weights_sort = torch.randn(2*self.dim*self.n + 1 , self.dim, requires_grad=True)*1#normal distribution, maybe uiform will work better
          self.weights_dot = torch.randn(2*self.dim*self.n + 1 , (self.n - self.dim), requires_grad=True)*1 #projections part of matrix
        elif not self.unique and self.original:
          self.weights_sort = torch.randn(2*self.dim_orig*self.n_orig + 1 , self.dim, requires_grad=True)*1 #normal distribution, maybe uiform will work better
          self.weights_dot = torch.randn(2*self.dim_orig*self.n_orig + 1 , self.n, requires_grad=True)*1 #entire matrix
        ############################################################################################################
        ######################################matrix multiplications################################################
        ############################################################################################################
        if not self.original:
            gram, projections = torch.split(mat, [self.dim, (self.n - self.dim)], dim=1)
            gram = gram.transpose(1,0)
            gram  = torch.reshape(gram,(-1,)) #linear map each d-dim vector one after the other
            sort_matmul = torch.matmul(torch.t(projections), torch.t(self.weights_sort)) #shape : n x 2nd+1, complexity: dxnx(2nd+1)
            column_wise_sort, indices = torch.sort(sort_matmul, dim = 0) # complexity: (2nd+1)(nlog(n)), dim direction to sort along 
            dot_matmul = torch.matmul(self.weights_dot, column_wise_sort) # shape: (2nd+1)^2 , complexity (2nd+1)x n x (2nd+1) = O(d^2 x n^3)
            embed = torch.diagonal(dot_matmul)
            embed = torch.cat([gram,embed])
        #recieve embedding 
        if self.original:
            sort_matmul = torch.matmul(torch.t(mat), torch.t(self.weights_sort)) #shape : n x 2nd+1, complexity: dxnx(2nd+1)
            column_wise_sort, indices = torch.sort(sort_matmul, dim = 0) # complexity: (2nd+1)(nlog(n)), dim direction to sort along 
            dot_matmul = torch.matmul(self.weights_dot, column_wise_sort) #shape: (2nd+1)^2 , complexity (2nd+1)x n x (2nd+1) = O(d^2 x n^3)
            embed = torch.diagonal(dot_matmul)
        return embed
class Vectorize(nn.Module):
  def __init__(self,dimension, n) -> None:
      super(Vectorize, self).__init__()
      self.dim = dimension
      self.n = n
  def combinations(self, matrix: torch.tensor) -> torch.tensor:
      all_idx_list = [item for item in range(self.n)] 
      combs = list(it.permutations(all_idx_list, r = self.dim))#list of all index combinations, non-repeating
      assert( len(combs) ==  factorial(self.n)//factorial(self.n-self.dim))
      vec = torch.empty(factorial(self.n)//factorial(self.n-self.dim), (self.dim**2 + (2*self.n*self.dim + 1)))
      for idx, item in enumerate(combs):
        torch.manual_seed(42)
        ug = UpperGram(self.dim, self.n, list(item))
        torch.manual_seed(42)
        ug = ug.forward(matrix)
        torch.manual_seed(42)
        embed = Embed(self.dim, self.n, unique=False)
        torch.manual_seed(42)
        embed = embed.forward(ug).clone()
        vec[idx, : ] = embed
      return vec.t()
  def forward(self, matrix : torch.tensor) -> torch.tensor:
      torch.manual_seed(42)
      embed = Embed(  (self.dim**2 + (2*self.n*self.dim + 1)), (factorial(self.n)//factorial(self.n-self.dim)), dim_orig=self.dim, n_orig=self.n, unique=False) # can be improved self.n ->self.n-dim
      torch.manual_seed(42)
      mat = self.combinations(matrix)
      embed = embed.forward(mat)
      return embed
    
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


        label = data[1].to(device)
        label = label.squeeze().to(device)
        #device = label.get_device()
        #print(str(device))
        data_now = torch.squeeze(data[0]).to(device)
        pred = model(data_now).to(device)
        #pred = torch.clamp(pred, min=-5, max=6).to(device)
        classify =  1 if pred[1] > pred[0] else 0

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
        label = 1 if label[1] > label[0] else 0
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
        print("cuda enabled")
    else:
        device = torch.device('cpu')
    torch.cuda.device(device)
    epochs = 3
    #test_interval = 10

    batch_size = 1 #SGD
    cloud_1 = torch.clone(cloud1).to(device)
    cloud_2 = torch.clone(cloud2).to(device)

    train_len = 500
    test_len = 100
    dataset_train = DatasetGen( train_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce = True )
    dataset_test = DatasetGen( test_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce = True )
    dataset_eval = DatasetGen( test_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce= True )

    dataloader_train = DataLoader( dataset_train,  batch_size=batch_size, shuffle=True)
    dataloader_eval = DataLoader( dataset_eval,  batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader( dataset_test,  batch_size=batch_size, shuffle=True)

    dataloaders = { 'train' : dataloader_train, 'test' : dataloader_test, 'valid' : dataloader_eval}
    torch.manual_seed(42)
    output_size = 2*dimension*n + 1
    softmax_dim = 2
    layer_sizes = torch.mul(torch.tensor([2, 1, 1]), output_size)
    model = nn.Sequential(
        Vectorize(dimension=dimension, n=n),
        nn.Linear(output_size, layer_sizes[0]),
        nn.ReLU(),
        nn.Linear(layer_sizes[0], layer_sizes[1]),
        nn.ReLU(),
        nn.Linear(layer_sizes[1], softmax_dim),
        nn.Softmax()
    )
    lr = 2e-4
    wd = 1e-6
    #iterations = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    loss_fun = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    res = {'epochs': [], 'losess': [], 'accuracies': [],'best_val': 1e-10, 'best_test': 1e-10, 'best_epoch': 0}

    for epoch in range(0, epochs):
        _, _, loss_arr,accuracy_list = train(epoch, dataloaders['train'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'train')
        torch.save(accuracy_list, 'degen_tensor_train_acc_{}.pt'.format(epoch))
        torch.save(loss_arr, 'degen_tensor_train_loss_{}.pt'.format(epoch))
        val_loss, val_acc, _, _ = train(epoch, dataloaders['valid'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'valid')
        test_loss, test_acc, _, _ = train(epoch, dataloaders['test'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'test')
        res['epochs'].append(epoch)
        res['accuracies'].append(test_acc)
        if val_acc > res['best_val']:
            res['best_val'] = val_acc
            res['best_test'] = test_acc
            res['best_epoch'] = epoch
        print("Val accurcy: %.4f \t test accuracy: %.4f \t epoch %d" % (val_acc, test_acc, epoch))
        print("Val accuracy: %.4f \t test accuracy: %.4f \t epoch %d" % (val_acc, test_acc, epoch))
        print("Best: val accuracy: %.4f \t test accuracy: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
    return res['best_test']

if __name__ == "__main__":
    res2 = {'GramOd' :  {  'gortler_3' : None, 'gortler_4' : None }}
    points = {  'gortler_3' : (a,b, 6,6), 'gortler_4' : (gortler_4_a, gortler_4_b,8,8)  }
    for key, value in res2.items():
        for key2, value2 in value.items():
            res2[key][key2] = main(points[key2][0], points[key2][1],points[key2][2],points[key2][3])
            f = open("/home/snirhordan/GramNN/gortler.pkl","wb")

            # write the python object (dict) to pickle file
            pickle.dump(res2,f)

            # close file
            f.close()
            print(res2)
