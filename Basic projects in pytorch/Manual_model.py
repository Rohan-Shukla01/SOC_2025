import numpy as np
#f=w*x
#f=2*x
x=np.array([1,2,3,4],dtype=np.float32)
y=np.array([2,4,6,8],dtype=np.float32)
w=0.0
#model prediction 
def forward(x):
    return w*x
#loss
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()
#gradient
#MSE=1/N*(w*x-y)**2
#dJ/dw=1/N *(w*x-y)*2*x
def gradient(x,y,yp):
    return 2*np.mean((yp-y)*x)
    
print(f'Prediction before training : f(5) = {forward(5):.3f}')
#training
learning_rate=0.03
n_iters=15
for  epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    #loss
    l=loss(y,y_pred)
    #gradients
    dw=gradient(x,y,y_pred)
    #updating wights
    w=w-(learning_rate*dw)
    print(f'epoch {epoch+1}: w= {w:.3f}, loss={l:.8f}')
print(f'Prediction after training : f(5) = {forward(5):.3f}')