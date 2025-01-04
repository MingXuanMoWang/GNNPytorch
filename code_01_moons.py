import sklearn.datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from code_02_moons_fun import LogicNet, plot_losses,plot_decision_boundary,predict
from sklearn.metrics import accuracy_score

np.random.seed(0)
X, Y = sklearn.datasets.make_moons(200,noise=0.2)
arg = np.squeeze(np.argwhere(Y==0),axis=1)
arg2 = np.squeeze(np.argwhere(Y==1),axis=1)
plt.title("moons coradata")
plt.scatter(X[arg,0],X[arg,1],s=100,c='b',marker='+',label='data1')
plt.scatter(X[arg2,0],X[arg2,1],s=40,c='r',marker='o',label='data2')
plt.legend()
plt.show()

model = LogicNet(inputdim=2,hiddendim=3,outputdim=2)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

xt = torch.from_numpy(X).type(torch.FloatTensor)
yt = torch.from_numpy(Y).type(torch.LongTensor)
epochs = 1000
losses = []
for i in range(epochs):
    loss = model.getloss(xt, yt)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plot_losses(losses)
print(accuracy_score(model.predict(xt),yt))
plot_decision_boundary(lambda x : predict(model,x),xt.numpy(),yt.numpy())