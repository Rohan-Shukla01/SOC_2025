#this is an effort to implement Vanilla RNN model
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
class RNN(nn.Module):
    def __init__(self,in_size,out_size,hidden_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.model=nn.Sequential(
            nn.Linear(in_size+hidden_size,hidden_size),
            nn.Tanh()
        )
        self.out=nn.Linear(hidden_size,out_size)
        self.h=torch.zeros(self.hidden_size)
    def forward(self,x,h=None,cat='train'):
        if h is None:
            h=self.h
        for i in x.float():
            a=torch.cat((i,h),dim=-1)
            h=self.model(a)
        y=self.out(h)
        if (cat=='train'):
            self.h=h
        return y,h
    def get_loss(self,y_hat,y):
        loss=nn.CrossEntropyLoss()
        return loss(y_hat,y)
in_size,out_size,hidden_size=5,5,9
num_ex=35
num_sent=10
num_epochs=20
char_embed={"a":[1,0,0,0,0],"b":[0,1,0,0,0],"c":[0,0,1,0,0],"d":[0,0,0,1,0],"e":[0,0,0,0,1]}
l=sorted(list(char_embed.keys()))
rules={
    "a":["c"], "b":["d"],"c":["a"],"d":["b"], "e":["a"]
}
x_list=[random.choice(l) for j in range(num_sent)]
for j in range(num_sent):
    for i in range(num_ex):
        x_list[j]+=random.choice(rules[x_list[j][-1]])
print(x_list)
target_list=[torch.tensor([l.index(i) for i in x_list[j]],dtype=torch.long) for j in range(num_sent)]

x_embed_list=[torch.zeros((len(x_list[j]),in_size)) for j in range(len(x_list))]
for j in range(len(x_list)):
    for i in range(len(x_list[j])):
        x_embed_list[j][i][:]=torch.tensor(char_embed[x_list[j][i]],dtype=float)
model=RNN(in_size,out_size,hidden_size)
optimizer=torch.optim.Adam(model.parameters(),lr=0.007)
loss_list=[]
for j in range(num_epochs):
    #h=torch.zeros(hidden_size)
    m=[]
    for k in range(len(x_list)):
        h=torch.zeros(hidden_size)
        for i in range(len(x_list[k])-1):
            y_hat,h=model(x_embed_list[k][:i+1],h)
            y=target_list[k][i+1]
            optimizer.zero_grad()
            loss=model.get_loss(y_hat.view(-1,out_size),torch.tensor([y]))
            loss.backward()
            optimizer.step()
            h=h.detach()
            m.append(loss)
    loss_list.append(torch.mean(torch.tensor(m)))       
    #print(torch.softmax(y_hat,dim=-1),' ',loss.item())
acc=[]
for j in range(len(x_embed_list)):
    str=""
    counter=0

    with torch.no_grad():
        for i in range(len(x_embed_list[j])):
            y,h=model(x_embed_list[j][:i+1],None,'test')
            k=int(torch.argmax(y))
            str=str+l[k]
    
    for i in range(len(x_list[j])-1):
        if x_list[j][i+1]==str[i]:
            counter+=1
    acc.append(float(counter)/(len(str)-1))
print(torch.mean(torch.tensor(acc),dtype=float).item())
plt.plot(range(1,num_epochs+1),loss_list,color='r')
plt.show()