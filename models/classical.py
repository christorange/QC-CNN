import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

n_class = 3
n_features = 196

class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 4, 2, stride = 2)
        # self.lr1 = nn.LeakyReLU(0.1)
        # self.ln1 = nn.LayerNorm(32, elementwise_affine=True)
        self.fc1 = nn.Linear(4*7*7, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, X):
        bs = X.shape[0]
        X = X.view(X.shape[0], 1, 14, 14)
        X = self.conv(X)
        X = F.relu(X)
        X = X.view(bs,-1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        return X
