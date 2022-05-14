import pennylane as qml
import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)

n_qubits = 4
n_layers = 1
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
        # side_len = X.shape[2] - self.kernel_size + 1  # *******
        x_lst = []
        for i in range(0, x.shape[2]-1,2):
            for j in range(0, x.shape[3]-1,2):
                x_lst.append(self.ql1(torch.flatten(x[:, :, i:i + self.kernel_size, j:j + self.kernel_size], start_dim=1)))
        x = torch.cat(x_lst,dim=1).view(bs,4,7,7)
        return x

class Inception(nn.Module):
    def __init__(self,in_channels):
        super(Inception, self).__init__()

        self.branchClassic_1 = nn.Conv2d(in_channels,4,kernel_size=1,stride=1)
        self.branchClassic_2 = nn.Conv2d(4,8,kernel_size=4,stride=2,padding=1)

        self.branchQuantum = Quanv2d(kernel_size=2)

    def forward(self,x):
        classic = self.branchClassic_1(x)
        classic = self.branchClassic_2(classic)

        quantum = self.branchQuantum(x)

        outputs = [classic,quantum]
        return torch.cat(outputs,dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.incep = Inception(in_channels=1)
        self.fc1 = nn.Linear(12*7*7,32)
        self.fc2 = nn.Linear(32,10)
        self.lr = nn.LeakyReLU(0.1)

    def forward(self,x):
        bs = x.shape[0]
        x = x.view(bs,1,14,14)
        x = self.incep(x)
        x = self.lr(x)

        x = x.view(bs,-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x