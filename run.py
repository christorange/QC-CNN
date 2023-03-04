import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from app.load_data import MyCSVDatasetReader as CSVDataset
from models.classical import Net
#from models.single_encoding import Net
#from models.multi_encoding import Net
#from models.hybrid_layer import Net
#from models.inception import Net
#from models.multi_noisy import Net
from app.train import train_network

# load the dataset
dataset = CSVDataset('./datasets/mnist_179_1200.csv')
# output location/file names
# outdir = 'results_255_tr_mnist358'
# file_prefix = 'mnist_358'


# load the device
device = torch.device('cpu')

# define model
net = Net()
# net.to(device)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adagrad(net.parameters(), lr = 0.5) # optimizer

epochs = 10
bs = 30

train_id, val_id = train_test_split(list(range(len(dataset))), test_size = 0.2, random_state = 0)
train_set = Subset(dataset, train_id)
val_set = Subset(dataset, val_id)


train_network(net = net, train_set = train_set, val_set = val_set, device = device, 
epochs = epochs, bs = bs, optimizer = optimizer, criterion = criterion)  # outdir = outdir, file_prefix = file_prefix)
