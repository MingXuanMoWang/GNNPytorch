import sklearn.datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
# from code_02_moons_fun import LogicNet, plot_losses,plot_decision_boundary,predict
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