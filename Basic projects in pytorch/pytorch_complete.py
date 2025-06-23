import torch
import torch.nn as nn
#f=w*x
#f=2*x
x=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
z=torch.tensor([5],dtype=torch.float32)
#we need  x and y to be shaped so that the outer array conttains all training sets while the inner ones contain the features
#namual weight not required
n_sample,n_features=x.shape
print(n_sample,n_features)
input_size=n_features
output_size=1
learning_rate=0.06
n_iters=15
#manual prediction not required
#model=nn.Linear(input_size,output_size)
#another way to write the linear func
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.lin=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        return self.lin(x)
model=LinearRegression(input_size,output_size)
#manual loss no longer required
loss=nn.MSELoss() #mean squared error
optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)   #Stochastic Gradient Descent
#manually computed gradient no longer needed
    
print(f'Prediction before training : f(5) = {model(z).item():.3f}')
#training

for  epoch in range(n_iters):
    #prediction=forward pass
    y_pred=model(x)
    #loss
    l=loss(y,y_pred)
    #gradients = back pass
    l.backward()
    #manual optimization no longer required
    optimizer.step()
    optimizer.zero_grad()
    [w,b]=model.parameters()
    print(f'epoch {epoch+1}: w= {w.item():.3f}, loss={l:.8f}')
print(f'Prediction after training : f(5) = {model(z).item():.3f}')