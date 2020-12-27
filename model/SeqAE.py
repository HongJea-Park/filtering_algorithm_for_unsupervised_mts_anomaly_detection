import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.torch_utils import torch_device
from utils.seq2chunk import seq2chunk


class Dataset(Dataset):
    """
    Pytorch Dataset instance for SeqAE.
    """
    def __init__(self, data: np.ndarray, filtering: bool = False,
                 y_pred: np.ndarray = None, subsequence_len: int = 30,
                 time_step: int = 10):

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

    def __getitem__(self, idx: int):
        return self.data[idx], self.data[idx, 1:]

    def get_label(self, label: np.ndarray):
        label = seq2chunk(data=label,
                          win_size=self.win_size,
                          overlap_size=self.overlap_size)
        if self.filtering:
            label = label[self.chunk_pred]
        return label.reshape(-1, self.win_size)[:, -1]


class Encoder(nn.Module):
    def __init__(self, num_feature, hidden_dim):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(num_feature, hidden_dim, 1,
                           bidirectional=False, batch_first=True)

    def forward(self, inputs):
        self.rnn.flatten_parameters()
        x, (hidden, cell) = self.rnn(inputs)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, num_feature, hidden_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(num_feature, hidden_dim, 1,
                           bidirectional=False, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_feature)

    def forward(self, inputs, hidden, cell):
        self.rnn.flatten_parameters()
        inputs = inputs.unsqueeze(1)
        output, (hidden, cell) = self.rnn(inputs, (hidden, cell))
        output = self.fc_out(output.squeeze())
        return output, hidden, cell


class Model(nn.Module):
    """
    Sequence-to-Sequence Auto-Encoder class instance.
    """
    def __init__(self, num_feature, hidden_dim=128):
        super(Model, self).__init__()

        self.encoder = Encoder(num_feature, hidden_dim)
        self.decoder = Decoder(num_feature, hidden_dim)

    def forward(self, inputs, target, device, tf_ratio=0.5):
        output = inputs[:, 0, :]
        batch_size, target_len, num_feature = inputs[:, 1:, :].shape

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, num_feature).to(device)
        hidden, cell = self.encoder(inputs)

        for t in range(target_len):
            output, hidden, cell = self.decoder(output, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < tf_ratio
            output = inputs[:, t+1, :] if teacher_force else output

        return outputs


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
        output = model(X, target, device)
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
            output = model(X, target, device, 0)
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
        output = model(X, target, device, 0)
        s = _anomaly_score(output, target).cpu().detach().numpy()
        score = np.concatenate((score, s))

    return score


def _loss_LSTMAE(recon_x, x):
    return nn.MSELoss(reduction='sum')(recon_x, x)


def _anomaly_score(recon_x, x):
    return nn.MSELoss(reduction='none')(recon_x, x).sum(axis=(1, 2))
