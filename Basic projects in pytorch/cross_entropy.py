import numpy as np

def Cross_entropy(y,y_hat):
    loss=-np.mean(np.sum((y*np.log(y_hat)),axis=1),axis=0)
    return loss
Y=np.array([1,0,0])
Y_good_pred=np.array([[0.7,0.2,0.1]])
Y_bad_pred=np.array([[0.1,0.3,0.6]])
l1=Cross_entropy(Y,Y_good_pred)
l2=Cross_entropy(Y,Y_bad_pred)
print(" good pred :",l1)
print(" bad pred : ",l2)

#Torch version
import torch
import torch.nn as nn
loss=nn.CrossEntropyLoss()
#size =nsamples x nclasses=3 X 3
Y= torch.tensor([0,2,1])# definig that Y belongs to class 0 i.e. tha first class
Y_pred_good=torch.tensor([[2.0,1.0,0.1],[3.0,1.0,5.0],[0.1,3.0,0.5]])
Y_pred_bad=torch.tensor([[0.2,1.0,1.0],[10.0,0.1,0.1],[5.0,1.0,0.5]])
l1=loss(Y_pred_good,Y)
l2=loss(Y_pred_bad,Y)
print("good: ",l1)
print("bad : ",l2)
_,predictions1=torch.max(Y_pred_good,1)
_,predictions2=torch.max(Y_pred_bad,1)
print(predictions1,'\n',predictions2)