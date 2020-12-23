import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.torch_utils import torch_device
from utils.seq2chunk import seq2chunk


class Dataset(Dataset):
    """
    Pytorch Dataset instance for MSCRED.
    Output is subsequence of signature matrix and signature matrix
    at last timestamp.
    """
    def __init__(self, data: np.ndarray, filtering: bool = False,
                 y_pred: np.ndarray = None, attention_size: int = 5,
                 time_step: int = 3, w_list: list = [10, 20, 30]):

        self.data = data
        self.data_len = self.data.shape[0]
        self.filtering = filtering
        self.w_list = sorted(w_list)
        self.attention_size = attention_size
        self.win_size = self.attention_size + self.w_list[-1] - 1
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

        # convert from chunk data to signature matrix at each timestamp
        self.n_f = self.data.shape[2]
        self.n_w = len(self.w_list)
        self.data = self.data.reshape(self.data.shape[0], -1)
        self.data = np.apply_along_axis(
            func1d=lambda x: self._get_signature_matrix(x),
            axis=1,
            arr=self.data)

        # convert to pytorch tensor from numpy array
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return self.data[idx], self.data[idx, -1]

    def _get_signature_matrix(self, a: np.ndarray):
        a = a.reshape(self.win_size, self.n_f)
        a = seq2chunk(data=a,
                      win_size=self.w_list[-1],
                      overlap_size=self.w_list[-1]-1)
        s_mat = np.zeros((self.attention_size, self.n_w, self.n_f, self.n_f))
        for t in range(self.attention_size):
            for i, w in enumerate(self.w_list):
                s_mat[t, i] = np.matmul(a[t, -w:].T, a[t, -w:])/w

        return s_mat

    def get_label(self, label):
        assert self.data_len == label.shape[0],\
            "label length should be equal to data length"
        label = seq2chunk(data=label,
                          win_size=self.win_size,
                          overlap_size=self.overlap_size)
        if self.filtering:
            label = label[self.chunk_pred]
        return (label.sum(axis=1) > 0).astype(int).reshape(-1)


class ConvLSTMCell(nn.Module):
    """
    Initialize ConvLSTM cell.

    Parameters
    ----------
    input_dim: int
        Number of channels of input tensor.
    hidden_dim: int
        Number of channels of hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel.
    bias: bool
        Whether or not to add the bias.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: tuple,
                 bias: bool):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv,
                                             self.hidden_dim,
                                             dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width,
                            device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width,
                            device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM implemented pytorch.

    Parameters
    ----------
        input_dim: int
            Number of channels in input
        hidden_dim: int
            Number of hidden channels
        kernel_size: (int, int)
            Size of kernel in convolutions
        num_layers: int
            Number of LSTM layers stacked on each other
        batch_first: bool
            Whether or not dimension 0 is the batch or not
        bias: bool
            Bias or no bias in Convolution
        return_all_layers: bool
            Return the list of computations for all layers

        Note: Will do same padding.

    Input
    ----------
        input_tensor: torch.Tensor
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)

    Output:
    ----------
        A tuple of two lists of length num_layers
        (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of
                lists of length T of each output
            1 - last_state_list is the list of last states
                each element of the list is a tuple (h, c)
                for hidden state and memory

    Example:
    ----------
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, (3, 3), 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: tuple,
                 num_layers: int, batch_first: bool = False, bias: bool = True,
                 return_all_layers: bool = False):
        super(ConvLSTM, self).__init__()
        # Make sure that both `kernel_size` and `hidden_dim` are lists
        # having len == num_layers
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        assert len(kernel_size) == len(hidden_dim) == num_layers,\
            'Inconsistent list length.'

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor: torch.Tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: torch.Tensor
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: torch.Tensor
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size: int, image_size: tuple):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size: int):
        assert (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and
                all([isinstance(elem, tuple) for elem in kernel_size]))),\
            '`kernel_size` must be tuple or list of tuples'

    @staticmethod
    def _extend_for_multilayer(param: list, num_layers: int):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Model(nn.Module):
    """
    MSCRED model class instance.
    """
    # pytorch dimension order is batch-channel-height-width
    def __init__(self, num_win: int = 3, num_feature: int = 25):
        super(Model, self).__init__()

        self.num_win = num_win
        self.num_feature = num_feature

        # calculate feature map size and deconvolutional layer kernel size
        self.f_size2 = np.ceil(self.num_feature/2).astype(int)
        self.f_size3 = np.ceil(self.f_size2/2).astype(int)
        self.f_size4 = np.ceil(self.f_size3/2).astype(int)
        self.k_size2 = self._get_kernel_size(self.f_size2, self.num_feature)
        self.k_size3 = self._get_kernel_size(self.f_size3, self.f_size2)
        self.k_size4 = self._get_kernel_size(self.f_size4, self.f_size3)

        # convolutional encoding layer
        # output_size = (input_size - kernel_size + 2*padding) / strides + 1
        self.conv1 = _conv_encoder(self.num_win, 32, (3, 3), 1)
        self.conv2 = _conv_encoder(32, 64, (3, 3), 2)
        self.conv3 = _conv_encoder(64, 128, (3, 3), 2)
        self.conv4 = _conv_encoder(128, 256, (3, 3), 2)

        # convLSTM
        self.convlstm1 = ConvLSTM(32, 32, (3, 3), 1, True, True, False)
        self.convlstm2 = ConvLSTM(64, 64, (3, 3), 1, True, True, False)
        self.convlstm3 = ConvLSTM(128, 128, (3, 3), 1, True, True, False)
        self.convlstm4 = ConvLSTM(256, 256, (3, 3), 1, True, True, False)

        # deconvolutional decoding layer
        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.deconv1 = _conv_decoder(64, 3, (3, 3), 1, 1)
        self.deconv2 = _conv_decoder(128, 32, (self.k_size2, self.k_size2), 2)
        self.deconv3 = _conv_decoder(256, 64, (self.k_size3, self.k_size3), 2)
        self.deconv4 = _conv_decoder(256, 128, (self.k_size4, self.k_size4), 2)

    def _get_kernel_size(self, in_size: int, out_size: int, stride: int = 2):
        return out_size - stride*in_size + stride + 2

    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
            X: torch.Tensor
                [B, T, W, N, N]
                B: Batch size
                T: Time size
                W: The number of windows
                N: The number of features
        """
        B, T, *S = X.shape

        # reshape input tensor to encode with convolutional layers
        X = X.reshape(-1, *S)

        # encoding
        e1 = self.conv1(X)  # [B*T, C1, N1, N1], C: The number of channels
        e2 = self.conv2(e1)  # [B*T, C2, N2, N2]
        e3 = self.conv3(e2)  # [B*T, C3, N3, N3]
        e4 = self.conv4(e3)  # [B*T, C4, N4, N4]

        # reshape encoded feature maps
        e1 = e1.reshape(B, T, *e1.shape[1:])  # [B, T, C1, N1, N1]
        e2 = e2.reshape(B, T, *e2.shape[1:])  # [B, T, C1, N1, N1]
        e3 = e3.reshape(B, T, *e3.shape[1:])  # [B, T, C1, N1, N1]
        e4 = e4.reshape(B, T, *e4.shape[1:])  # [B, T, C1, N1, N1]

        # convLSTM
        output1, (hidden1, _) = self.convlstm1(e1)
        output2, (hidden2, _) = self.convlstm2(e2)
        output3, (hidden3, _) = self.convlstm3(e3)
        output4, (hidden4, _) = self.convlstm4(e4)

        # attention mechanism in first layer
        size = output1.shape[2:]
        hidden = hidden1.reshape(B, -1)
        output = output1.reshape(B, T, -1)
        feature_map1 = _attention(hidden, output, output).reshape(-1, *size)

        # attention mechanism in second layer
        size = output2.shape[2:]
        hidden = hidden2.reshape(B, -1)
        output = output2.reshape(B, T, -1)
        feature_map2 = _attention(hidden, output, output).reshape(-1, *size)

        # attention mechanism in third layer
        size = output3.shape[2:]
        hidden = hidden3.reshape(B, -1)
        output = output3.reshape(B, T, -1)
        feature_map3 = _attention(hidden, output, output).reshape(-1, *size)

        # attention mechanism in fourth layer
        size = output4.shape[2:]
        hidden = hidden4.reshape(B, -1)
        output = output4.reshape(B, T, -1)
        feature_map4 = _attention(hidden, output, output).reshape(-1, *size)

        # decoding
        d4 = torch.cat((feature_map3, self.deconv4(feature_map4)), dim=1)
        d3 = torch.cat((feature_map2, self.deconv3(d4)), dim=1)
        d2 = torch.cat((feature_map1, self.deconv2(d3)), dim=1)
        d1 = self.deconv1(d2)

        return d1


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
        loss = _loss_MSCRED(output, target)
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
    model.eval()

    with torch.no_grad():
        for X, target in valid_loader:
            batch_size = X.size(0)
            num_data += batch_size
            X, target = X.to(device), target.to(device)
            output = model(X)
            s = _anomaly_score(output, target).cpu().detach().numpy()
            score = np.concatenate((score, s))
            loss = _loss_MSCRED(output, target)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / num_data

    return avg_valid_loss, score


def get_score(model: nn.Module,
              data_loader: torch.utils.data.dataloader.DataLoader):
    score = np.array([])
    device = torch_device(model)
    model.eval()

    with torch.no_grad():
        for X, target in data_loader:
            X, target = X.to(device), target.to(device)
            output = model(X)
            s = _anomaly_score(output, target).cpu().detach().numpy()
            score = np.concatenate((score, s))

    return score


def _frobenius_norm(recon_smat: torch.Tensor, smat: torch.Tensor):
    assert recon_smat.dim() == 4 and smat.dim() == 4,\
        "input and output should be 4-D tensor"
    return torch.norm(smat - recon_smat, p='fro', dim=(-1, -2)).sum(axis=1)


def _loss_MSCRED(recon_smat: torch.Tensor, smat: torch.Tensor):
    assert recon_smat.dim() == 4 and smat.dim() == 4,\
        "input and output should be 4-D tensor"
    return _frobenius_norm(recon_smat, smat).sum()


def _anomaly_score(recon_smat: torch.Tensor, smat: torch.Tensor):
    return _frobenius_norm(recon_smat, smat)


def _conv_encoder(in_channels: int, out_channels: int, kernel_size: tuple,
                  stride: int, padding: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding),
        nn.ReLU())


def _conv_decoder(in_channels: int, out_channels: int, kernel_size: tuple,
                  stride: int, padding: int = 1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding),
        nn.ReLU())


def _attention(query: torch.Tensor, keys: torch.Tensor,  values: torch.Tensor,
               rescale: int = 5):
    """
    A function to apply attention mechanism.

    Parameters
    ----------
    query: torch.Tensor
        [B, vector_size)]
    keys: torch.Tensor
        [B, T, vector_size)]
    values: torch.Tensor
        [B, T, vector_size)]
    """
    query = query.unsqueeze(1)  # [B, vector_size] -> [B, 1, vector_size]
    keys = keys.permute(0, 2, 1)  # [B, T, vector_size] -> [B, vector_size, T]
    score = torch.bmm(query, keys)  # [B, 1, T]
    weight = F.softmax(score.mul_(1/rescale), dim=2)
    a_score = torch.bmm(weight, values).squeeze(1)  # [B, vector_size]
    return a_score
