import numpy as np
import torch
def softmax(x):
    return np.exp(x)/sum(np.exp(x))
x=np.array([2.0,1.0,0.1])
print(softmax(x))
y=torch.tensor([2.0,1.0,0.1])
z=torch.softmax(y,dim=0) #dim parameter here asks for the dimension
                         #to compute softmax onto
                         #entries along dim sum up to 1

print(z)