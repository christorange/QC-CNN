import pennylane as qml
from math import ceil, pi
import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)

n_qubits = 4
n_layers = 2
dev = qml.device('default.qubit', wires=n_qubits)

def circuit(inputs, weights):
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)
        # qml.RY(inputs[qub], wires=qub)

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[layer,i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits,2*n_qubits):
            qml.RY(weights[layer,j], wires=j % n_qubits)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class Quanv2d(nn.Module):
    def __init__(self, kernel_size):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers,2*n_qubits)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method="best")
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        #self.stride = stride

    def forward(self, x):
        assert len(x.shape) == 4  # (bs, c, w, h)
        bs = x.shape[0]
        c = x.shape[1]
        # side_len = X.shape[2] - self.kernel_size + 1  # *******
        x_lst = []
        for i in range(0, x.shape[2]-1,2):
            for j in range(0, x.shape[3]-1,2):
                x_lst.append(self.ql1(torch.flatten(x[:, :, i:i + self.kernel_size, j:j + self.kernel_size], start_dim=1)))
        x = torch.cat(x_lst,dim=1)  # .view(bs,n_qubits,14,14)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.qc = Quanv2d(kernel_size=2)
        # self.conv = nn.Conv2d(4,32,4)
        self.fc1 = nn.Linear(4*7*7,20)
        self.fc2 = nn.Linear(20,10)
        # self.pooling = nn.MaxPool2d(2)

    def forward(self,x):
        bs = x.shape[0]
        x = x.view(bs,1,14,14)
        x = self.qc(x)
        # x = F.relu(x)
        # x = x.view(bs,-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


