from torch_scatter import scatter_mean
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

import torch 
import numpy as np

class QM9Data(Dataset):#zpve label
    def __init__(self, fold='train') -> None:
        super().__init__()
        self.dataset = QM9(root='/home/snirhordan/GramNN/data')
        self.all_instances = []
        self.all_labels = []
        self.fold = fold
        self.populate()
    def populate(self):
        for data in self.dataset:
            self.all_instances.append( torch.vstack((torch.t(data.pos), data.z)))
            self.all_labels.append(data.y.squeeze()[6]) #zpve
        self.length = len(self.all_labels)
        self.train_length = torch.ceil(torch.tensor(self.length) * 0.8)
    def populate(self):
        for idx, data in enumerate(self.dataset):
            self.all_instances.append( torch.vstack((torch.t(torch.tensor(data, dtype=torch.double)), self.z)))
    def __getitem__(self, idx):
        if self.fold == 'train':
            return self.all_instances[idx], self.all_labels[idx]
        if self.fold == 'test':
            return self.all_instances[self.train_length + idx], self.all_labels[self.train_length + idx]
    def __len__(self):
        if self.fold == 'train':
            return torch.tensor(self.train_length, dtype=torch.int32)
        if self.fold == 'test':
            return torch.tensosr(self.length - self.train_length, dtype=torch.int32)
class Dimer(Dataset):
    def __init__(self, fold='train') -> None:
        super().__init__()
        self.dataset = torch.load('/home/snirhordan/GramNN/positions.pt') #distance invariance, not dot product invariance
        self.all_instances = []
        self.all_labels = torch.load('/home/snirhordan/GramNN/labels.pt')
        self.length = len(self.all_labels.tolist())
        self.train_length = torch.ceil(torch.tensor(self.length) * 0.8)
        self.fold = fold
        self.z = torch.tensor([8,1,1,8,1,1], dtype=torch.float32)
        self.populate()
    def populate(self):
        for data in self.dataset:
            self.all_instances.append( torch.vstack((torch.t(torch.clone(data)).to(dtype=torch.float32), self.z)))
    def __getitem__(self, idx):
        if self.fold == 'train':
            return self.all_instances[idx], self.all_labels[idx]
        if self.fold == 'test':
            return self.all_instances[self.train_length + idx], self.all_labels[self.train_length + idx]
    def __len__(self):
        if self.fold == 'train':
            return torch.tensor(self.train_length, dtype=torch.int32)
        if self.fold == 'test':
            return torch.tensosr(self.length - self.train_length, dtype=torch.int32)


class SOAP(Dataset):
    def __init__(self, fold='train') -> None:
        super().__init__()
        self.dataset = torch.load('/home/snirhordan/GramNN/dataset.pt') #distance invariance, not dot product invariance
        self.all_labels = torch.load('/home/snirhordan/GramNN/labels.pt')
        self.length = len(self.all_labels.tolist())
        self.train_length = torch.ceil(torch.tensor(self.length) * 0.8)
        self.fold = fold
    def __getitem__(self, idx):
        if self.fold == 'train':
            return self.dataset[idx], self.all_labels[idx]
        if self.fold == 'test':
            return self.dataset[self.train_length + idx], self.all_labels[self.train_length + idx]
    def __len__(self):
        if self.fold == 'train':
            return torch.tensor(self.train_length, dtype=torch.int32)
        if self.fold == 'test':
            return torch.tensor(self.length - self.train_length, dtype=torch.int32)