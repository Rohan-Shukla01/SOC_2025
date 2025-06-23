import torch
import torch.nn as nn #for mseloss and and sigmoid
import numpy as np
from sklearn import datasets #for dataset
from sklearn.preprocessing import StandardScaler #for z-score scaling
from sklearn.model_selection import train_test_split as tts
# a) prepare data
data=datasets.load_breast_cancer()
z,y=data.data,data.target
n_samples,n_features=z.shape
print(n_samples,n_features)
x_train,x_test,y_train,y_test=tts(z,y,random_state=5,test_size=0.2)
#scale 
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(-1,1)
y_test=y_test.view(-1,1)

# 1) model
#f=wx+b, sigmoid function as the end
class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super().__init__()
        self.lin=nn.Linear(n_input_features,1)
    def forward(self,x):
        y_pred=torch.sigmoid(self.lin(x))
        return y_pred
model=LogisticRegression(n_features)
#loss and optim
learning_rate=0.01
criterion=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
# training loop
num_epochs=1000
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted=model.forward(x_train)
    loss=criterion(y_predicted,y_train)
    #backward pass
    loss.backward()
    #update
    optimizer.step()
    optimizer.zero_grad()
    if (epoch%100==0):
        print(f'epoch no.: {epoch+1},loss :{loss.item():.4f}')
#evaluation
with torch.no_grad():
    y_pred=model.forward(x_test)
    y_predicted_cls=y_pred.round()
    acc=y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f"{acc:.4f}")





