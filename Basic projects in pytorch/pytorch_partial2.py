#Steps to make a model
# 1) Design model (input,output size,forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#        -)forward pass: compute prediction
#        -)backward pass: gradients
#        -)update weights 
import torch
import torch.nn as nn
#f=w*x
#f=2*x
x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
learning_rate=0.03
n_iters=15
#model prediction 
def forward(x):
    return w*x #works as w is 1 dimensional, otherwise there would've been an error
#manual loss no longer required
loss=nn.MSELoss() #mean squared error
optimizer= torch.optim.SGD([w],lr=learning_rate)   #Stochastic Gradient Descent
#manually computed gradient no longer needed
    
print(f'Prediction before training : f(5) = {forward(5):.3f}')
#training

for  epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    #loss
    l=loss(y,y_pred)
    #gradients = back pass
    l.backward()
    #manual optimization no longer required
    optimizer.step()
    optimizer.zero_grad()
    print(f'epoch {epoch+1}: w= {w:.3f}, loss={l:.8f}')
print(f'Prediction after training : f(5) = {forward(5):.3f}')