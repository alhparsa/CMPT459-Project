from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn
import torch
import tqdm


"""
Trained with CrossEntropy loss and model_1 and model_2 use network_1 architecture
Model_3 and model_4 use nework_2 achitecture.

Parameters:
model_1 and model_2 were trained with SGD optimizer and lr of 0.01 and 0.001 repesectively
model_3 and model_4 were trained with Adam optimizer and lr of 0.01 and 0.001 repesectively

Epochs:
model_1 and model_2 were trained for 20 and 35 epochs.
model_3 and model_4 were trained for 10 and 15 epochs.
"""

X_train, y_train = [], []
X_val, y_val = [], []

class train_data(Dataset):
    def __init__(self, train):
        self.train = train

    def __len__(self):
        if self.train:
            global X_train
            return len(X_train)
        else:
            global X_val
            return len(X_val)

    def __getitem__(self, idx):
        if self.train:
            global X_train
            global y_train
            item = torch.tensor(X_train[idx], dtype=torch.float, requires_grad=True)
            label = torch.tensor(y_train[idx], dtype=torch.int8)
        else:
            global X_val
            global y_val
            item = torch.tensor(X_val[idx], dtype=torch.float, requires_grad=False)
            label = torch.tensor(y_val[idx], dtype=torch.int8)
        return item, label
        
        #         return torch.tensor(np.array(self.data[idx], dtype='float'), requires_grad=True), torch.tensor(self.label[idx], requires_grad=False).long()


class network(torch.nn.Module):
    def __init__(self, first_layer_size=50, second_layer_size=250, activation = 'relu'):
        super(network, self).__init__()
        self.linear1 = torch.nn.Linear(13, first_layer_size)
        self.linear2 = torch.nn.Linear(first_layer_size, second_layer_size)
        self.output = torch.nn.Linear(second_layer_size, 4)
        self.activation = getattr(torch, activation)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.output(x)
        return x


class network_2(torch.nn.Module):
    def __init__(self):
        super(network_2, self).__init__()
        self.linear1 = torch.nn.Linear(13, 50)
        self.output = torch.nn.Linear(50, 4)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.output(x)
        return x


def eval(model, X, y):
    return ((torch.argmax(torch.softmax(model(X), dim=1), dim=1) == y).sum() / len(y)*100).item()


def load_model(model, pth):
    model.load_state_dict(torch.load(pth))


def load_data(X_t, y_t, batch_size=250, num_workers=0, train=True):
    if train:
        global X_train
        global y_train
        X_train = X_t.to_numpy()
        y_train = y_t.to_numpy()
    else:
        global X_val
        global y_val
        X_val = X_t.to_numpy()
        y_val = y_t.to_numpy()
    return torch.utils.data.DataLoader(train_data(train), batch_size=batch_size, num_workers=num_workers)

def train_model(model, X_t, y_t, epochs=10, lr=0.01, path='model_1.pth'):
    dataloader = load_data(X_t, y_t)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = epochs
    total_loss = []
    for i in tqdm.tqdm(range(epochs)):
        ls = 0
        for i_batch, sample_batched in enumerate(dataloader):
            X, y = sample_batched
            y_hat = model(X.float())
            loss = criterion(y_hat, y.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ls += loss.cpu().float()
        total_loss.append(ls/(i_batch+1.))
        print(f'iter: {i}, loss: {ls/(i_batch+1.)}')
    torch.save(model.state_dict(), path)
