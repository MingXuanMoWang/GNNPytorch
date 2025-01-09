import torch
from torch import nn
from code_22_WGAN import device,displayAndTest,train,WGAN_G,WGAN_D

class CondWGAN_D(WGAN_D):
    def __init__(self,inputch=2):
        super(CondWGAN_D,self).__init__(inputch)
        self.labfc1 = nn.Linear(10,28*28)
    def forward(self,x,lab):
        d_in = torch.cat((x.view(x.size(0),-1),self.labfc1(lab)),-1)
        x = d_in.view(d_in.size(0),2,28,28)
        return super(CondWGAN_D,self).forward(x,lab)
class CondWGAN_G(WGAN_G):
    def __init__(self,input_size,input_n=2):
        super(CondWGAN_G,self).__init__(input_size,input_n)
        self.labfc1 = nn.Linear(10,input_size)
    def forward(self,x,lab):
        d_in = torch.cat((x,self.labfc1(lab)),-1)
        return super(CondWGAN_G,self).forward(d_in,lab)
if __name__ == '__main__':
    z_dimension = 40
    D = CondWGAN_D().to(device)
    G = CondWGAN_G(z_dimension).to(device)
    train(D,G,'./condw_img',z_dimension)
    displayAndTest(D,G,z_dimension)

