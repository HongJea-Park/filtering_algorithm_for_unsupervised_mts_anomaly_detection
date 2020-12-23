import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.torch_utils import torch_device
from utils.seq2chunk import seq2chunk


class Dataset(Dataset):
    """
    Pytorch Dataset instance for LSTM.
    """
    def __init__(self, data: np.ndarray, filtering: bool = False,
                 y_pred: np.ndarray = None, subsequence_len: int = 30,
                 time_step: int = 1):

        self.data = data
        self.data_len = self.data.shape[0]
        self.filtering = filtering
        self.win_size = subsequence_len + 1
        self.overlap_size = self.win_size - time_step

        # convert from 2d sequence data and pred
        # to 3d chunk data and chunk pred if pred array exist
        self.data = seq2chunk(data=self.data,
                              win_size=self.win_size,
                              overlap_size=self.overlap_size)
        self.n_seq = self.data.shape[0]
        self.n_f_seq = self.data.shape[0]
        if self.filtering:
            self.y_pred = seq2chunk(data=y_pred,
                                    win_size=self.win_size,
                                    overlap_size=self.overlap_size)

            # remain chunk data containing only normal data
            self.chunk_pred = (self.y_pred.sum(axis=1) == 0).reshape(-1)
            self.data = self.data[self.chunk_pred]
            self.n_f_seq = self.data.shape[0]

        # convert to pytorch tensor from numpy array
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]

    def get_label(self, label):
        label = seq2chunk(data=label,
                          win_size=self.win_size,
                          overlap_size=self.overlap_size)
        if self.filtering:
            label = label[self.chunk_pred]
        return (label.sum(axis=1) > 0).astype(int).reshape(-1)


class Model(nn.Module):
    """
    LSTM predictor class instance.
    """
    def __init__(self, num_feature: int = 25, hidden_dim: int = 128):
        super(Model, self).__init__()
        self.num_feature = num_feature
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=self.num_feature,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True)
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.num_feature)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.layernorm(x)
        x = self.linear(x)

        return x


def train(model: nn.Module,
          train_loader: torch.utils.data.dataloader.DataLoader,
          optimizer: torch.optim, epoch: int):
    train_loss = 0
    train_loss_list = []
    batch_list = []
    num_data = 0
    device = torch_device(model)
    model.train()

    for X, target in train_loader:
        batch_size = X.size(0)
        num_data += batch_size
        X, target = X.to(device), target.to(device)
        output = model(X)
        loss = _loss_LSTMAE(output, target)
        train_loss += loss.item()
        train_loss_list.append(loss.item())
        batch_list.append(epoch-1 + (num_data / len(train_loader.sampler)))
        # backpropagation and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / num_data

    return avg_train_loss, train_loss_list, batch_list


def valid(model: nn.Module,
          valid_loader: torch.utils.data.dataloader.DataLoader):
    valid_loss = 0
    num_data = 0
    score = np.array([])
    device = torch_device(model)

    with torch.no_grad():
        for X, target in valid_loader:
            batch_size = X.size(0)
            num_data += batch_size
            X, target = X.to(device), target.to(device)
            output = model(X)
            s = _anomaly_score(output, target).cpu().detach().numpy()
            score = np.concatenate((score, s))
            loss = _loss_LSTMAE(output, target)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / num_data

    return avg_valid_loss, score


def get_score(model: nn.Module,
              data_loader: torch.utils.data.dataloader.DataLoader):
    score = np.array([])
    device = torch_device(model)
    model.eval()
    for X, target in data_loader:
        X, target = X.to(device), target.to(device)
        output = model(X)
        s = _anomaly_score(output, target).cpu().detach().numpy()
        score = np.concatenate((score, s))

    return score


def _loss_LSTMAE(recon_x, x):
    return nn.MSELoss()(recon_x, x)


def _anomaly_score(recon_x, x):
    return nn.MSELoss(reduction='none')(recon_x, x).sum(axis=(1, 2))
