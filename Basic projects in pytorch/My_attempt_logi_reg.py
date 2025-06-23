import torch
import torch.nn as nn
import numpy as np
import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
'''def Z_score(x):
    
    std=np.std(x,axis=0)
    mean=np.mean(x,axis=0)
    for i in range(x.shape[1]):
        x[:,i]=(x[:,i]-mean[i])/std[i]
    
    return x,mean,std'''
#setting up data:
data=datasets.make_classification(n_samples=1000,n_features=5,n_classes=2,random_state=5)
z,y=data
x_train,x_test,y_train,y_test=tts(z,y,test_size=0.2,random_state=5)
Scaler=StandardScaler() #this initailization is necessary as we want to fit bot sets with the same standardscaler object
x_train=Scaler.fit_transform(x_train)
x_test=Scaler.transform(x_test)
n_features=x_train.shape[1]
n_samples=x_train.shape[0]
print(type(x_test),type(x_train),type(y_test),type(y_train))
x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32)).view(-1,1)
y_train=torch.from_numpy(y_train.astype(np.float32)).view(-1,1)
learning_rate=0.03
lambda_=0.01
#model setup
model=nn.Linear(x_train.shape[1],1)
def forward(x):
    return torch.sigmoid(model(x))
loss=nn.BCELoss()
def loss1(y_pred,y):
    l1=loss(y_pred,y)
    w=model.weight
    
    l1+=lambda_*(w[0].dot(w[0]))/n_samples
    return l1
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
num_iters=10000
for i in range(num_iters):
    y_pred=forward(x_train)
    l=loss1(y_pred,y_train)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (i%1000==0):
        print(f'epoch no. {i+1}, loss : {l}')
with torch.no_grad():
    y_pred=forward(x_test)
    y_pred=y_pred.round()
    acc=y_pred.eq(y_test).sum()/float(len(y_test))
    print(f' acc: {acc:.4f}')