import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math
#we use dataset wine,with the first column being the class label
#and all other columns are features
class WineDataset(Dataset):
    def __init__(self,transform=None):

        #data loading
        xy=np.loadtxt(r"C:\Users\Rohan\Documents\my prog journey\SOC_2025\Grad descent with autograd and pytorch\wine.csv",delimiter=',',dtype=np.float32,skiprows=1)
        #skiprows parameter to tell the number of rows to skip, here we skip the header
        self.x=xy[:,1:]
        self.y=xy[:,[0]] #[0] has to be specified to get ndarray from y[index]
        self.transform=transform
        self.n_samples=xy.shape[0]

    def __getitem__(self, index):
        sample=self.x[index],self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return sample
        # datadet [0] ->allow for indexing
    def __len__(self):
        #allow for len(dataset)
        return self.n_samples
#now, lets create custom transform classes
class ToTensor:
    def __call__(self,sample):
        inputs,targets=sample
        return torch.from_numpy(inputs),torch.from_numpy(targets)
class MulTransform:
    def __init__(self,factor):
        self.factor=factor
    def __call__(self,sample):
        inputs,labels=sample
        inputs*=self.factor
        return inputs,labels
dataset=WineDataset(transform=None)
first_data=dataset[0]
features,labels=first_data
print(type(features),' ',type(labels))
print(features)
composed=torchvision.transforms.Compose([ToTensor(),MulTransform(2)])
#this takes a list of transforms in order to apply and returns a comosite transform
dataset=WineDataset(transform=composed)
first_data=dataset[0]
features,labels=first_data
print(type(features),' ',type(labels))
print(features)
