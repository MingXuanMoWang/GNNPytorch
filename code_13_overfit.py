import sklearn.datasets
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from code_02_moons_fun import LogicNet, plot_losses,plot_decision_boundary,predict, moving_average
from sklearn.metrics import accuracy_score

np.random.seed(0)
X, Y = sklearn.datasets.make_moons(40,noise=0.2)
arg = np.squeeze(np.argwhere(Y==0),axis=1)
arg2 = np.squeeze(np.argwhere(Y==1),axis=1)
plt.title("moons coradata")
plt.scatter(X[arg,0],X[arg,1],s=100,c='b',marker='+',label='data1')
plt.scatter(X[arg2,0],X[arg2,1],s=40,c='r',marker='o',label='data2')
plt.legend()
plt.show()

class Logic_Dropout_Net(LogicNet):
    def __init__(self,inputdim,hiddendim,outputdim):
        super(Logic_Dropout_Net,self).__init__(inputdim,hiddendim,outputdim)
        self.BN = nn.BatchNorm1d(hiddendim)
    def forward(self,x):
        x = self.Linear1(x)
        x = torch.tanh(x)
        # x = nn.functional.dropout(x,p = 0.07,training=self.training)
        x = self.BN(x)
        x = self.Linear2(x)
        return x

model = Logic_Dropout_Net(inputdim=2,hiddendim=500,outputdim=2)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# weight_p,bias_p = [],[]
# for name, p in model.named_parameters():
#     if 'bias' in name:
#         bias_p += [p]
#     else:
#         weight_p += [p]
# optimizer = torch.optim.Adam([{'params':weight_p,'weight_decay':0.001},
#                               {'params':bias_p,'weight_decay':0}],
#                              lr=0.01)

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
avgloss = moving_average(losses)
plt.figure(1)
plt.subplot(211)
plt.plot(range(len(avgloss)),avgloss,'b--')
plt.xlabel('step number')
plt.ylabel('Training loss')
plt.title('step number vs. Training loss')
plt.show()

plot_decision_boundary(lambda x : predict(model,x),X,Y)
print("训练时的准确度：",accuracy_score(model.predict(xt),yt))

Xtest, Ytest = sklearn.datasets.make_moons(80,noise=0.2)
plot_decision_boundary(lambda x: predict(model,x),Xtest,Ytest)
Xtest_t = torch.from_numpy(Xtest).type(torch.FloatTensor)
Ytest_t = torch.from_numpy(Ytest).type(torch.LongTensor)
print("测试时的准确率：",accuracy_score(model.predict(Xtest_t),Ytest_t))