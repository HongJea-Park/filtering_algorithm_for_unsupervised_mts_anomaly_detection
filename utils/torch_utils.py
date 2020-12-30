import torch
import numpy as np
from collections import OrderedDict
from joblib import dump, load

from utils.mkdir import mkdir


class Checkpoint():
    """
    Class instance to save checkpoint for training log and model.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.checkpoint_dir = mkdir(f'./checkpoint/{model_name}/')

        self.log_dir = f'{self.checkpoint_dir}/log.log'
        self.state_dir = f'{self.checkpoint_dir}/state.tar'
        self.model_dir = f'{self.checkpoint_dir}/model.pth'
        self.anomaly_score_dir = f'{self.checkpoint_dir}/anomaly_score.npz'

        self.batch_list = []
        self.epoch_list = []
        self.train_loss_list_per_batch = []
        self.train_loss_list_per_epoch = []
        self.valid_loss_list = []

    def save_log(self, batch_list: list, epoch: int,
                 train_loss_list_per_batch: list, train_loss_per_epoch: float,
                 valid_loss: float):
        self.batch_list.extend(batch_list)
        self.epoch_list.append(epoch)
        self.train_loss_list_per_batch.extend(train_loss_list_per_batch)
        self.train_loss_list_per_epoch.append(train_loss_per_epoch)
        self.valid_loss_list.append(valid_loss)

        self.num_batch = len(self.batch_list) // len(self.epoch_list)

        log = {'batch_list': self.batch_list,
               'epoch': self.epoch_list,
               'train_loss_per_batch': self.train_loss_list_per_batch,
               'train_loss_per_epoch': self.train_loss_list_per_epoch,
               'valid_loss': self.valid_loss_list}

        torch.save(log, self.log_dir)

    def load_log(self, return_best: bool = False):

        print(f"\n loading log {self.log_dir}'")

        log = torch.load(self.log_dir)

        self.batch_list = log['batch_list']
        self.epoch_list = log['epoch']
        self.train_loss_list_per_batch = log['train_loss_per_batch']
        self.train_loss_list_per_epoch = log['train_loss_per_epoch']
        self.valid_loss_list = log['valid_loss']

        if return_best:
            best_valid_loss = np.min(self.valid_loss_list)
            return self.epoch_list[-1] + 1, best_valid_loss

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim,
                        is_best: bool, save_state: bool = True):
        if save_state:
            state = {'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, self.state_dir)

        if is_best:
            torch.save(model.state_dict(), self.model_dir)

    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim):
        state = torch.load(self.state_dir)
        try:
            model.load_state_dict(state['model_state_dict'])
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in state['model_state_dict'].items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        return model, optimizer

    def load_model(self, model: torch.nn.Module):
        model_state = torch.load(self.model_dir)
        try:
            model.load_state_dict(model_state)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        return model

    def save_anomaly_score(self, anomaly_label: np.ndarray,
                           anomaly_score: np.ndarray,
                           filtered_score: np.ndarray):
        np.savez_compressed(
            file=self.anomaly_score_dir,
            allow_pickle=True,
            anomaly_label=anomaly_label,
            anomaly_score=anomaly_score,
            filtered_score=filtered_score)

    def load_anomaly_score(self):
        loader = np.load(self.anomaly_score_dir, allow_pickle=True)
        anomaly_label = loader['anomaly_label']
        anomaly_score = loader['anomaly_score']
        filtered_score = loader['filtered_score']
        return anomaly_label, anomaly_score, filtered_score


class Early_stopping():
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.is_best = False

    def __call__(self, score: float, curses, stdscr, lower_best: bool = True):
        if self.best_score is None:
            self.best_score = score
            self.is_best = True
        elif lower_best:
            if score > self.best_score + self.delta:
                self.counter += 1
                self.is_best = False
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
                self.is_best = True
        else:
            if score < self.best_score + self.delta:
                self.counter += 1
                self.is_best = False
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
                self.is_best = True

        if self.patience != np.inf:
            stdscr.addstr(14, 0, 'EarlyStopping: ',
                          curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(
                '>'*self.counter+'|'*(self.patience-self.counter)+'| ',
                curses.color_pair(2) | curses.A_BOLD)

        return self.early_stop, self.is_best


def torch_device(model):

    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def update_epoch(epoch: int, n_seq: int, n_f_seq: int, batch_size: int):
    if epoch == np.inf:
        return epoch
    total_update = int(np.ceil(n_seq / batch_size)) * epoch
    num_update_per_epoch = int(np.ceil(n_f_seq / batch_size))
    num_update = num_update_per_epoch * epoch
    epoch += int((total_update - num_update) / num_update_per_epoch)
    return epoch
