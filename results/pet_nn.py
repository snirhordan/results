
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
        #self.addOrthogonal()
        #self.addGaussian(0, 0.)
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
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
C = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1,1,0],
    [-1,-1,0],
    [1,2,0],
    [-1,2,0],
    [0,0,1]
], dtype=torch.double,device=device)

D = torch.tensor([
    [-2,0,-2],
    [2,0,2],
    [1,1,0],
    [-1,-1,0],
    [1,2,0],
    [-1,2,0],
    [0,0,-1]
], dtype=torch.double,device=device)
#first one from Pod. 2020 that egnn should separate
A = torch.tensor([[-2,0,-2],\
                  [2,0,2],\
                  [0,1,1],\
                  [1,1,0],\
                  [-1,-1,0] ], dtype=torch.double, device=device)

B = torch.tensor([[-2,0,-2],\
                  [2,0,2],\
                  [0,1,-1],\
                  [1,1,0],\
                  [-1,-1,0] ], dtype=torch.double, device=device)

# Incompleteness 2020 https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.125.166001/si.pdf


import torch
from torch import nn
from scipy import stats

class EGNNProject(nn.Module):
    """"
    Implementation of embedding with EGNN(C=0)-level expressivity 

    Input is b x d x n point cloud
    sparse1,2 means using scipy for distribution
    delta is about applying exponential(-x/delta)

    """
    def __init__(self, dim=3, n=6,const=2, batch_size=16, delta = 1., exp=False):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.delta = delta
        self.seed = 42
        self.dim = dim
        self.n = n
        self.batch_size = batch_size
        self.const = const #const of embedding const*d*n +1
        self.embed_dim = self.const*n*dim + 1
        self.dtype = torch.double
        self.exp=exp
        torch.manual_seed(42)
        self.w = nn.Linear(self.n, self.embed_dim, device=self.device, dtype=torch.double)
        self.w.weight.retain_grad()
    #    self._init_weights(self.w)
        torch.manual_seed(42)
        self.W = nn.Linear(self.n, self.embed_dim, device=self.device, dtype=torch.double)
        self.W.weight.retain_grad()
    #    self._init_weights(self.W)
        self.to(self.device, dtype=torch.double)
    #def _init_weights(self, module):
    #    if isinstance(module, nn.Linear):
    #        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
    #        if module.bias is not None:
    #            module.bias.data.zero_()
    def forward(self, cloud):
        """
        Input is a b x d x n point cloud
        """
        #### create distance matrix + norms on diagonal
        cloud = cloud.transpose(dim0=-1, dim1=-2) # shape b x n x d
        dist_zero_diag = torch.cdist(cloud, cloud, p=2)
        norms = torch.linalg.norm(cloud, ord=None, axis=2, keepdims=False) #checked correct
        #### sort distances
        egnn_style, _ = torch.sort(dist_zero_diag, dim=1)
        ### add norms
        egnn_style[:,0, :] = norms #matrix with sorted distances from each node and node norm unsorted at top, size b x n x n
        if self.exp:
          egnn_style = torch.exp(torch.div((-1)*egnn_style, self.delta)) #exponential scaling
        #### project on 2nd+1 space
        #apply embedding
        sorted, _ = torch.sort( self.w(egnn_style.transpose(dim0=-1,dim1=-2)), dim=1)
        ## matrix multiplication
        embed = self.W(sorted.transpose(dim0=-1,dim1=-2))
        embed = torch.diagonal(embed, offset=0)
        return embed.t()
def train(epoch, loader, model, loss_fun, lr_scheduler, optimizer, batch_size=16,device='cuda:0', partition = 'train'):
    torch.manual_seed(42)

    res = {'loss': 0, 'counter': 0, 'loss_arr':[], 'accuracy': 0, 'acc_list' : []}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train() 
            optimizer.zero_grad()

        else:
            model.eval()

        #get data
        label = data[1].to(device)
        label = label.to(device)
        data_now = data[0].to(device, dtype=torch.double)
        #predict
        pred = model(data_now.cuda())

        if partition == 'train':

            loss = loss_fun(pred.to(torch.float).cuda(), label.to(torch.float).cuda())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        else:
            loss = loss_fun(pred.to(torch.float), label.to(torch.float))
        
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition
        log_interval = 10
        if i % log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    return res['loss'] / res['counter'], res['loss_arr']

def main(cloud1, cloud2, dimension,n):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.device(device)
    epochs = 50
    const=10
    #test_interval = 10

    batch_size = 1 #SGD
    cloud_1 = torch.clone(cloud1).to(device)
    cloud_2 = torch.clone(cloud2).to(device)

    train_len = 100000
    test_len = 1000
    dataset_train = DatasetGen( train_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce = True, device=device )
    dataset_test = DatasetGen( test_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce = True, device=device )
    dataset_eval = DatasetGen( test_len, dimension, n, torch.t(cloud_1), torch.t(cloud_2), ce= True, device=device )

    dataloader_train = DataLoader( dataset_train,  batch_size=batch_size, shuffle=True)
    dataloader_eval = DataLoader( dataset_eval,  batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader( dataset_test,  batch_size=batch_size, shuffle=True)

    dataloaders = { 'train' : dataloader_train, 'test' : dataloader_test, 'valid' : dataloader_eval}
    torch.manual_seed(42)
    output_size = const*dimension*n + 1
    softmax_dim = 2
    layer_sizes = torch.mul(torch.tensor([4, 2, 1], device=device), output_size)
    model = nn.Sequential(
        EGNNProject(dim=dimension, n=n, batch_size=batch_size,  const=const, exp=False, delta=1.), #const must match output size
        nn.Linear(output_size, layer_sizes[0], dtype=torch.double),
        nn.ReLU(),
        nn.Linear(layer_sizes[0], layer_sizes[1], dtype=torch.double),
        nn.ReLU(),
        nn.Linear(layer_sizes[1], 20, dtype=torch.double),
        nn.ReLU(),
        nn.Linear(20, softmax_dim, dtype=torch.double),
        nn.Softmax()
    ).to(device)
    lr = 8e-5
    wd = 1e-6
    #iterations = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    loss_fun = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    for epoch in range(0, epochs):
        _, _ = train(epoch, dataloaders['train'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'train')
        val_loss,  _ = train(epoch, dataloaders['valid'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'valid')
        test_loss , _ = train(epoch, dataloaders['test'], model, loss_fun, lr_scheduler, optimizer, device=device, partition = 'test')
        print("Epoch :{}, val loss: {}, test loss : {} ".format(epoch, val_loss, test_loss))
        if val_loss < res['best_val']:
            res['best_val'] = val_loss
            res['best_test'] = test_loss
            res['best_epoch'] = epoch
        print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
    return res['best_test']

if __name__ == "__main__":
    if torch.cuda.is_available():
        use_cuda=True
    else:
        use_cuda=False
    main(A, B, 3, 5)

    
