import torch
#f=w*x
#f=2*x
x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
#model prediction 
def forward(x):
    return w*x #works as w is 1 dimensional, otherwise there would've been an error
#loss
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

#manually computed gradient no longer needed
    
print(f'Prediction before training : f(5) = {forward(5):.3f}')
#training
learning_rate=0.03
n_iters=15
for  epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    #loss
    l=loss(y,y_pred)
    #gradients = back passs
    l.backward()
    dw=w.grad
    #updating wights
    with torch.no_grad():
        #w=w-(learning_rate*dw)
        w-=(learning_rate*dw)
        ''' -= is an inplace operator whereas w=w-.. creates a new
         tensor everytime, thus overwriting whatever we had stored in w. '''
    w.grad.zero_()
    print(f'epoch {epoch+1}: w= {w:.3f}, loss={l:.8f}')
print(f'Prediction after training : f(5) = {forward(5):.3f}')