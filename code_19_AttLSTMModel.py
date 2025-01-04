import torchvision
import torchvision.transforms as tranforms
import pylab
import torch
from matplotlib import pyplot as plt
import numpy as np


data_dir = './fashion_mnist/'
tranform = tranforms.Compose([tranforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(data_dir,train=True,transform=tranform,download=True)

print("训练数据集条数：",len(train_dataset))
val_dataset = torchvision.datasets.FashionMNIST(root=data_dir,train=False,transform=tranform)
print("测试数据集条数：",len(val_dataset))
im = train_dataset[0][0]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()
print("该图片的标签为：",train_dataset[0][1])

batch_size = 10
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

def imshow(img):
    print("图片形状：",np.shape(img))
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg,(1,2,0)))
classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot')
sample = iter(train_loader)
images, labels = sample.__next__()
print('样本形状：',np.shape(images))
print('样本标签：',labels)
imshow(torchvision.utils.make_grid(images,nrow=batch_size))
print(','.join('%5s' % classes[labels[j]] for j in range(len(images))))
class myLSTMNet(torch.nn.Module):
    def __init__(self,in_dim,hidden_dim,n_layer,n_class):
        super(myLSTMNet,self).__init__()
        self.lstm = torch.nn.LSTM(in_dim,hidden_dim,n_layer,batch_first=True)
        self.Linear = torch.nn.Linear(hidden_dim*28,n_class)
        self.attention = AttentionSeq(hidden_dim,hard=0.03)
    def forward(self,t):
        t, _ = self.lstm(t)
        t = self.attention(t)
        t = t.reshape(t.shape[0],-1)
        out = self.Linear(t)
        return out


class AttentionSeq(torch.nn.Module):
    def __init__(self, hidden_dim, hard=0):
        super(AttentionSeq, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hard = hard
    def forward(self, features, mean=False):
        batch_size, time_step, hidden_dim = features.size()
        weight = torch.nn.Tanh()(self.dense(features))

        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
        mask_idx = mask_idx.unsqueeze(-1).repeat(1, 1, hidden_dim)
        weight = torch.where(mask_idx == 1, weight, torch.full_like(mask_idx, (-2 ** 32 + 1)))
        weight = weight.transpose(2, 1)
        weight = torch.nn.Softmax(dim=2)(weight)
        if self.hard != 0:
            weight = torch.where(weight > self.hard, weight, torch.full_like(weight, 0))
        if mean:
            weight = weight.mean(dim=1)
            weight = weight.unsqueeze(1)
            weight = weight.repeat(1, hidden_dim, 1)
        weight = weight.transpose(2, 1)
        features_attention = weight * features
        return features_attention
network = myLSTMNet(28, 128, 2, 10)  # 图片大小是28x28
#指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
network.to(device)
print(network)#打印网络

criterion = torch.nn.CrossEntropyLoss()  #实例化损失函数类
optimizer = torch.optim.Adam(network.parameters(), lr=.01)

for epoch in range(2): #数据集迭代2次
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0): #循环取出批次数据
        inputs, labels = data
        inputs = inputs.squeeze(1)
        inputs, labels = inputs.to(device), labels.to(device) #
        optimizer.zero_grad()#清空之前的梯度
        outputs = network(inputs)
        loss = criterion(outputs, labels)#计算损失
        loss.backward()  #反向传播
        optimizer.step() #更新参数

        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0




print('Finished Training')


#使用模型
dataiter = iter(test_loader)
images, labels = dataiter.__next__()

inputs, labels = images.to(device), labels.to(device)


imshow(torchvision.utils.make_grid(images,nrow=batch_size))
print('真实标签: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
inputs = inputs.squeeze(1)
outputs = network(inputs)
_, predicted = torch.max(outputs, 1)


print('预测结果: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(len(images))))


#测试模型
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.squeeze(1)
        inputs, labels = images.to(device), labels.to(device)
        outputs = network(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.to(device)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


sumacc = 0
for i in range(10):
    Accuracy = 100 * class_correct[i] / class_total[i]
    print('Accuracy of %5s : %2d %%' % (classes[i], Accuracy ))
    sumacc =sumacc+Accuracy
print('Accuracy of all : %2d %%' % ( sumacc/10. ))