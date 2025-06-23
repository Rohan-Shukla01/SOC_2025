import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
#0) make datasets
x_numpy,y_numpy=datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=5)
x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
y=y.view(y.shape[0],1)
n_samples,n_features=x.shape
# 1) model
model=nn.Linear(n_features,1)

# 2)loss and optimizer
learning_rate=0.01
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

# 3) training loop
num_epochs=10000
for epoch in range(num_epochs):
    #forward pass and loss
    y_pred=model(x)
    loss=criterion(y_pred,y)
    #backward pass
    loss.backward()
    #update
    optimizer.step()
    optimizer.zero_grad()
    if (epoch%1000==0):
        print(f'epoch {epoch+1}, loss= {loss.item():.4f}')
print(list(model.parameters()))
#plot
predicted=model(x).detach().numpy()
plt.scatter(x_numpy,y_numpy,c='r',marker='x')
plt.plot(x_numpy,predicted,'b')
plt.show()
#print(model.parameters())
