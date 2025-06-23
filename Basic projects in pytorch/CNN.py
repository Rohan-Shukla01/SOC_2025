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
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#hyper parameters

num_classes=10
num_epochs=10
batch_size=100
learning_rate=0.05
#CIFAR10
classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
train_dataset=torchvision.datasets.CIFAR10(root=r"C:\Users\Rohan\Documents\my prog journey\SOC_2025\Grad descent with autograd and pytorch",train=True,transform=transforms.ToTensor(), download=True)
test_dataset=torchvision.datasets.CIFAR10(root=r"C:\Users\Rohan\Documents\my prog journey\SOC_2025\Grad descent with autograd and pytorch",train=False,transform=transforms.ToTensor(), download=False)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
examples=iter(train_loader)
samples,labels=next(examples)
print(samples.shape,labels.shape, sep=' ')
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
#plt.show()
class Conv_NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1=nn.Conv2d(3,6,5)
        # 3 is the input channel size , i.e our inputs have 3 color channels, so each feature also has a depth of 3
        # each pixel now has 3 values associated with it
        # 6 is the output channel size, there will be 6 different feature block, each of size 5x5, looking for different patterns
        self.pool=nn.MaxPool2d(2,2)
        self.Conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,80)
        self.fc3=nn.Linear(80,10)
        
    def forward(self,x):
        x=self.pool(F.relu(self.Conv1(x)))
        x=self.pool(F.relu(self.Conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x #soft max included in loss
        
               
model=Conv_NeuralNet()

#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

# training loop
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        #forward pass
        output=model.forward(images)
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
    n_class_samples=[0 for i in range(num_classes)]
    n_class_correct=[0 for i in range(num_classes)]

    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        #value, index
        _,predictions=torch.max(outputs,1)
        n_samples+=labels.shape[0]
        n_correct_pred+=(predictions == labels).sum().item()
        for i in range(batch_size):
            label=labels[i]
            pred=predictions[i]
            n_class_samples[label]+=1
            if (label==pred):
                n_class_correct[label]+=1


    acc=100.0* n_correct_pred/n_samples
    for i in range(num_classes):
        print(f'class accuracy : class - {classes[i]}, acc- {100.0*n_class_correct[i]/n_class_samples[i]}')
    print("accuracy of whole network : ",acc) 