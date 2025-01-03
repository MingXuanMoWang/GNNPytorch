import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy import stats
import pandas as pd
titanic_data = pd.read_csv('titanic3.csv')
titanic_data = pd.concat([titanic_data,pd.get_dummies(titanic_data['sex']),
                          pd.get_dummies(titanic_data['embarked'],prefix="embark"),
                          pd.get_dummies(titanic_data['pclass'],prefix="class")],axis=1)

titanic_data["age"] = titanic_data["age"].fillna(titanic_data["age"].mean())
titanic_data["fare"] = titanic_data["fare"].fillna(titanic_data["fare"].mean())
titanic_data = titanic_data.drop(['name','ticket','cabin','boat','body','home.dest','sex','embarked','pclass'],axis=1)
print(titanic_data.columns)

labels = titanic_data["survived"].to_numpy()
titanic_data = titanic_data.drop(['survived'],axis=1)
data = titanic_data.to_numpy()

feature_names = list(titanic_data)
np.random.seed(10)
train_indices = np.random.choice(len(labels),int(0.7 * len(labels)),replace=False)
test_indices = list(set(range(len(labels))) - set(train_indices))
train_features = data[train_indices].astype(float).astype(np.float32)
train_labels = labels[train_indices]
test_features = data[test_indices].astype(float).astype(np.float32)
test_labels = labels[test_indices]
print(len(test_indices))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
torch.manual_seed(0)
class ThreelinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12,12)
        self.mish1 = Mish()
        self.linear2 = nn.Linear(12,8)
        self.mish2 = Mish()
        self.linear3 = nn.Linear(8,2)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,x):
        lin1_out = self.linear1(x)
        out1 = self.mish1(lin1_out)
        out2 = self.mish2(self.linear2(out1))
        return self.softmax(self.linear3(out2))
    def getloss(self,x,y):
        y_pred = self.forward(x)
        loss = self.criterion(y_pred,y)
        return loss
if __name__ == '__main__':
    net = ThreelinearModel()
    num_epochs = 200
    optimizer = torch.optim.Adam(net.parameters(),lr=0.04)
    input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(train_labels)
    losses = []
    for epoch in range(num_epochs):
        loss = net.getloss(input_tensor,label_tensor)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print('Epoch {}/{} => Loss:{:.2f}'.format(epoch+1,num_epochs,loss.item()))
    os.makedirs('models',exist_ok=True)
    torch.save(net.state_dict(),'models/titanic_model.pt')

    from code_02_moons_fun import plot_losses
    plot_losses(losses)

    out_probs = net(input_tensor).detach().numpy()
    out_classes = np.argmax(out_probs,axis=1)
    print("Train Accuracy:",sum(out_classes == train_labels) / len(train_labels))

    test_input_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)
    out_probs = net(test_input_tensor).detach().numpy()
    out_classes = np.argmax(out_probs,axis=1)
    print("Test Accuracy:",sum(out_classes == test_labels) / len(test_labels))
