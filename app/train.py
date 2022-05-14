import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os

def train_network(net = None, train_set = None, val_set = None, device = None, 
epochs = 10, bs = 20, optimizer = None, criterion = None):  # outdir = None, file_prefix = None):

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True)

    net = net.to(device)

    tr_losses = []
    val_losses = []
    tr_accs = []
    val_accs = []

    for epoch in range(epochs):
        t1 = time.time()
        net.train()
        tr_loss = 0

        y_trues = []
        y_preds = []

        for i, sampled_batch in enumerate(train_loader):

            t2 = time.time()

            data = sampled_batch['feature']
            y = sampled_batch['label'].squeeze()

            data = data.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)

            data = data.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = net(data)

            loss = criterion(output,y)

            loss.backward()
            optimizer.step()
            #print(net.ql1.weights.grad)
            tr_loss = tr_loss + loss.data.cpu().numpy()

            y_trues += y.cpu().numpy().tolist()
            y_preds += output.data.cpu().numpy().argmax(axis=1).tolist()

            print('batch({}):{:.4f}'.format(i,time.time()-t2))

        tr_acc = accuracy_score(y_trues, y_preds)
        tr_accs.append(tr_acc)
        tr_loss = tr_loss/(i+1)
        tr_losses.append(tr_loss)

        cnf = confusion_matrix(y_trues, y_preds)
        print(cnf)

        print('Epoch:{}, TR_Loss: {:.4f}, TR_Acc: {:.4f}'.format(epoch, tr_loss, tr_acc))

        net.eval()
        val_loss = 0

        y_trues = []
        y_preds = []

        for i, sampled_batch in enumerate(val_loader):
            data = sampled_batch['feature']
            y = sampled_batch['label'].squeeze()

            data = data.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)

            data = data.to(device)
            y = y.to(device)

            with torch.no_grad():
                output = net(data)

            loss = criterion(output,y)
            val_loss = val_loss + loss.data.cpu().numpy()

            y_trues += y.cpu().numpy().tolist()
            y_preds += output.data.cpu().numpy().argmax(axis=1).tolist()

        val_acc = accuracy_score(y_trues, y_preds)
        val_accs.append(val_acc)
        val_loss = val_loss/(i+1)
        val_losses.append(val_loss)
        
        cnf = confusion_matrix(y_trues, y_preds)
        print(cnf)

        print('Epoch: {} VAL_Loss: {:.4f}, VAL_Acc: {:.4f}'.format(epoch, val_loss, val_acc))
        print('Time for Epoch ({}): {:.4f}'.format(epoch, time.time()-t1))
    
    #save model and results
    # os.makedirs(outdir, exist_ok = True)
    # torch.save(net.state_dict(), outdir + '/' + file_prefix + '_model')
    # np.save(outdir + '/' + file_prefix + '_training_loss.npy', tr_losses)
    # np.save(outdir + '/' + file_prefix + '_validation_loss.npy', val_losses)
    # np.save(outdir + '/' + file_prefix + '_training_accuracy.npy', tr_accs)
    # np.save(outdir + '/' + file_prefix + '_validation_accuracy.npy', val_accs)
