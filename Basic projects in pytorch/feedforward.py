#MNIST
# DataLoader , Transformation
#Multilayer Neural Net, activation function
#Loss and Optimizer
#Training Loop (batch training)
#Model evaluation
#gpu support
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#hyper parameters
input_size=784 #image size is 28x28 pixels
hidden_size=100 #try out diff sizes
num_classes=10
num_epochs=5
batch_size=100
learning_rate=0.01
#MNIST
train_dataset=torchvision.datasets.MNIST(root=r"C:\Users\Rohan\Documents\my prog journey\SOC_2025\Grad descent with autograd and pytorch",train=True,transform=transforms.ToTensor(), download=True)
test_dataset=torchvision.datasets.MNIST(root=r"C:\Users\Rohan\Documents\my prog journey\SOC_2025\Grad descent with autograd and pytorch",train=False,transform=transforms.ToTensor(), download=False)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
examples=iter(train_loader)
samples,labels=next(examples)
print(samples.shape,labels.shape, sep=' ')
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
#plt.show()
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.layer1=nn.Linear(in_features=input_size,out_features=hidden_size)
        self.relu=nn.ReLU()
        self.layer2=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        out=self.layer1(x)
        out=self.relu(out)
        out=self.layer2(out)
        #we will later use cross entropy loss, which will perform softmax
        return out
model=NeuralNet(input_size,hidden_size,num_classes)

#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# training loop
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        # 100 , 1, 28x28
        #input_size=784
        #so images must be reshaped to 784
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        #forward pass
        output=model(images)
        loss=criterion(output,labels)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if((i+1)%100==0):
            print(f'epoch : {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss= {loss.item():.4f}')
#test
with torch.no_grad():
    n_correct_pred=0
    n_samples=0
    for images,labels in test_loader:
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        outputs=model(images)
        #value, index
        _,predictions=torch.max(outputs,1)
        n_samples+=labels.shape[0]
        n_correct_pred+=(predictions == labels).sum().item()
    acc=100.0* n_correct_pred/n_samples
    print("accuracy : ",acc) 