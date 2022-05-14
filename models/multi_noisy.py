import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from math import ceil
from math import pi

torch.manual_seed(0)

n_qubits = 4
n_layers = 1
n_class = 3
n_features = 196
image_x_y_dim = 14
kernel_size = n_qubits
stride = 2

dev = qml.device("default.mixed", wires=n_qubits)


def circuit(inputs, weights):
    var_per_qubit = int(len(inputs) / n_qubits) + 1
    encoding_gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        for i in range(var_per_qubit):
            if (qub * var_per_qubit + i) < len(inputs):
                exec('qml.{}({}, wires = {})'.format(encoding_gates[i], inputs[qub * var_per_qubit + i], qub))
            else:  # load nothing
                pass

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[l, i], wires=[i, (i + 1) % n_qubits])
            # qml.CNOT(wires = [i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[l, j], wires=j % n_qubits)

    _expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return _expectations
    # return qml.expval(qml.PauliZ(0))


class Quanv2d(nn.Module):
    def __init__(self, kernel_size=None, stride=None):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers, 2 * n_qubits)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method='best')
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        assert len(X.shape) == 4  # (bs, c, w, h)
        XL = []
        for i in range(0, X.shape[2] - 2, stride):
            for j in range(0, X.shape[3] - 2, stride):
                XL.append(self.ql1(torch.flatten(X[:, :, i:i + kernel_size, j:j + kernel_size], start_dim=1)))
        X = torch.cat(XL, dim=1)
        return X


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.ql1 = Quanv2d(kernel_size=kernel_size, stride=stride)

        self.fc1 = nn.Linear(4 * 6 * 6, n_class * 2)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(n_class * 2, n_class)

    def forward(self, X):
        bs = X.shape[0]
        X = X.view(bs, 1, image_x_y_dim, image_x_y_dim)
        X = self.ql1(X)

        X = self.fc1(X)
        X = self.lr1(X)
        X = self.fc2(X)
        return X


