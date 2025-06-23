import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math
#we use dataset wine,with the first column being the class label
#and all other columns are features
class WineDataset(Dataset):
    def __init__(self):

        #data loading
        xy=np.loadtxt(r"C:\Users\Rohan\Documents\my prog journey\SOC_2025\Grad descent with autograd and pytorch\wine.csv",delimiter=',',dtype=np.float32,skiprows=1)
        #skiprows parameter to tell the number of rows to skip, here we skip the header
        self.x=torch.from_numpy(xy[:,1:])
        self.y=torch.from_numpy(xy[:,0])
        self.n_samples=xy.shape[0]

    def __getitem__(self, index):
        return self.x[index],self.y[index]
        # datadet [0] ->allow for indexing
    def __len__(self):
        #allow for len(dataset)
        return self.n_samples
dataset=WineDataset()
'''first_data=dataset[0]
features,labels=first_data
print(features,' ',labels)'''
dataloader=DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=0)
#the num_workers parrameter is to speed up the loading process by multiple internal processes.
#training lood -->
num_epochs=2
total_samples=len(dataset)
n_iterations=math.ceil(total_samples/4)
for epoch in range(num_epochs):
    for i,(inputs,labels) in enumerate(dataloader):
        # forward .backward,update
        if (i+1)%10==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

