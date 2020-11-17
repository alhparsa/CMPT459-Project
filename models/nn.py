from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn
import torch


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


class train_data(Dataset):
    def __init__(self, X_train, y_train):
        self.data = X_train
        self.label = y_train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(np.array(self.data.iloc[idx].values, dtype='float'), requires_grad=True), torch.tensor(self.label.iloc[idx], requires_grad=False).long()


class network_1(torch.nn.Module):
    def __init__(self):
        super(network_1, self).__init__()
        self.linear1 = torch.nn.Linear(13, 50)
        self.linear2 = torch.nn.Linear(50, 250)
        self.output = torch.nn.Linear(250, 4)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
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
    return (torch.argmax(torch.softmax(model(torch.tensor(X.values).float()), dim=1), dim=1).numpy() == y).sum() / len(y)*100


def load_model(model, pth):
    model.load_state_dict(torch.load(pth))


def load_data(train_dataset, batch_size=250, num_workers=0):
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
