import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from data.synthetic import Generator
from filtering.chunkfilter import Chunk_Filter
from utils import argtype
from utils.seq2chunk import seq2chunk
from utils.torch_utils import Checkpoint, Early_stopping, update_epoch
from vis.custom_vis import Custom_Vis


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
parser = argparse.ArgumentParser()

# about synthetic dataset
parser.add_argument('--anomaly_ratio',
                    type=float,
                    default=.05,
                    help='anomaly data ratio in train and valid set')
parser.add_argument('--anomaly_type',
                    type=str,
                    default='type1',
                    choices=['type1', 'type2'],
                    help='anomaly type of synthetic data')

# about chunk filter
parser.add_argument('--filtering',
                    type=argtype.boolean,
                    default='True',
                    help='filter or not')
parser.add_argument('--n_neighbors',
                    type=int,
                    default=20,
                    help='number of neighbors')
parser.add_argument('--iqr_multiplier',
                    type=float,
                    default=1.5,
                    help='constant to multiply iqr')
parser.add_argument('--filter_size',
                    type=int,
                    default=30,
                    help='chunk size for filtering')
parser.add_argument('--normalization',
                    type=argtype.boolean,
                    default='True',
                    help='data normalization during filtering process')

# about autoencoder
parser.add_argument('--model_name',
                    type=str,
                    default='MSCRED',
                    choices=['MSCRED', 'SeqAE', 'DeepAnT', 'LSTM'],
                    help='model name')
parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help='learning rate')
parser.add_argument('--batch_size',
                    type=int,
                    default=256,
                    help='batch size')
parser.add_argument('--epoch',
                    type=argtype.integer_or_inf,
                    default=200,
                    help='the number of epoch')
parser.add_argument('--patience',
                    type=argtype.integer_or_inf,
                    default='inf',
                    help='patience for early stopping')
parser.add_argument('--retrain',
                    type=argtype.boolean,
                    default='False',
                    help='retrain model or not')
parser.add_argument('--time_step',
                    type=int,
                    default=15,
                    help='sequence split step')
parser.add_argument('--gamma',
                    type=float,
                    default=0.99,
                    help='gamma for learning rate scheduler')

args = parser.parse_args()

if args.model_name == 'MSCRED':
    from model.MSCRED import *
elif args.model_name == 'SeqAE':
    from model.SeqAE import *
elif args.model_name == 'DeepAnT':
    from model.DeepAnT import *
elif args.model_name == 'LSTM':
    from model.LSTM import *


def main():
    # about synthetic data
    anomaly_ratio = args.anomaly_ratio
    anomaly_type = args.anomaly_type
    # about chunk filter
    filtering = args.filtering
    n_neighbors = args.n_neighbors
    iqr_multiplier = args.iqr_multiplier
    filter_size = args.filter_size
    normalization = args.normalization
    # about autoencoder
    model_name = args.model_name
    lr = args.lr
    batch_size = args.batch_size
    epoch = args.epoch
    patience = args.patience
    retrain = args.retrain
    time_step = args.time_step
    gamma = args.gamma

    random_state = 42
    data_len = 300000
    num_feature = 25
    t_idx = int(data_len * 0.3)
    v_idx = int(data_len * 0.5)

    data_name = \
        f'{anomaly_type}_ratio_{str(anomaly_ratio)[2:]:<03s}'
    model_name += f'_{data_name}_filtering_{filtering}'

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vis = Custom_Vis()

    # Generate synthetic data and split train, valid and test set
    generator = Generator(random_state=random_state)
    generator.generate(shape=(data_len, num_feature),
                       anomaly_ratio=anomaly_ratio,
                       split_ratio=[0.3, 0.2, 0.5],
                       min_feature_ratio=0.2,
                       max_feature_ratio=0.5,
                       min_len=20,
                       max_len=200,
                       anomaly_type=anomaly_type)
    data = generator.data.copy()
    label = generator.label.copy()
    ano_set = generator.ano_set
    ano_feature = generator.ano_features
    train_valid_data, train_valid_label = \
        data[:v_idx], label[:v_idx]
    test_data, test_label = data[v_idx:], label[v_idx:]

    transformed_data = MinMaxScaler().fit_transform(train_valid_data)
    chunk_data = seq2chunk(data=transformed_data, win_size=filter_size)

    # apply filtering or not
    if filtering:
        model_name += f'_iqr_multiplier{iqr_multiplier}'
        start = time.time()
        mts_filter = Chunk_Filter(data=chunk_data)
        filter_pred = mts_filter.fit(matrix_type='nng',
                                     n_neighbors=n_neighbors,
                                     iqr_multiplier=iqr_multiplier,
                                     normalization=normalization)
        # print filtering result
        acc, recall, precision, f1 = \
            mts_filter.get_metric(train_valid_label)
        filter_result = {'accuracy': acc,
                         'recall': recall,
                         'precision': precision,
                         'f1': f1,
                         'time': time.time() - start}
        vis.print_params(env=model_name,
                         params=filter_result,
                         title='Filtering Result',
                         clear=True)
        vis.lof_score(env=model_name,
                      lof_score=mts_filter.lof_score)
        clear = False
    else:
        iqr_multiplier = 0
        filter_pred = np.zeros(shape=train_valid_data.shape[0])
        clear = True

    print(f'data: {data_name} \n'
          f'model: {model_name} \n')

    train_data, train_label = \
        train_valid_data[:t_idx], train_valid_label[:t_idx]
    valid_data, valid_label = \
        train_valid_data[t_idx:], train_valid_label[t_idx:]
    train_pred = filter_pred[:t_idx]
    valid_pred = filter_pred[t_idx:v_idx]

    # normalize train data
    scaler = MinMaxScaler().fit(train_data[train_pred == 0])
    train_data = scaler.transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)

    print_params = {
        'data length': data_len,
        'anomaly ratio': anomaly_ratio,
        'anomaly type': anomaly_type,
        '# of anomaly in train set': int(train_label.sum()),
        '# of anomaly in validation set': int(valid_label.sum()),
        '# of anomaly in test set': int(test_label.sum()),
        '# of filtered in trainset': int(train_data[train_pred == 1].shape[0]),
        '# of filtered in validset': int(valid_data[valid_pred == 1].shape[0]),
        'filtering': filtering,
        'n_neighbors': n_neighbors,
        'train ratio': 0.3,
        'valid ratio': 0.2,
        'test ratio': 0.5
    }
    vis.print_params(env=model_name,
                     params=print_params,
                     title='Dataset Info',
                     clear=clear)

    vis.data_plot(env=f'{data_name}',
                  data=data,
                  ano_set=generator.ano_set,
                  ano_features=generator.ano_features,
                  clear=True)

    train_dataset = Dataset(data=train_data,
                            filtering=filtering,
                            y_pred=train_pred,
                            time_step=time_step)
    valid_dataset = Dataset(data=valid_data,
                            filtering=filtering,
                            y_pred=valid_pred,
                            time_step=time_step)
    valid_label = valid_dataset.get_label(valid_label)

    # # update epoch
    n_seq, n_f_seq = train_dataset.n_seq, train_dataset.n_f_seq
    epoch = update_epoch(epoch=epoch,
                         n_seq=n_seq,
                         n_f_seq=n_f_seq,
                         batch_size=batch_size)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True)

    if filtering:
        train_valid_data = np.concatenate((train_data, valid_data))
        filtered_dataset = Dataset(data=train_valid_data,
                                   filtering=filtering,
                                   y_pred=(filter_pred == 0),
                                   time_step=1)
        filtered_loader = DataLoader(dataset=filtered_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     pin_memory=True)

    model = Model(num_feature=num_feature).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)

    checkpoint = Checkpoint(model_name=model_name)
    early_stopping = Early_stopping(patience=patience)
    parameters = list(model.parameters())
    optimizer = Adam(parameters, lr=lr, weight_decay=1e-2)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    training_time = 0

    if retrain:
        model, optimizer = checkpoint.load_checkpoint(model, optimizer)
        checkpoint.load_log()
        e = checkpoint.epoch_list[-1]
    else:
        e = 0

    while e < epoch:
        e += 1
        batch_time = time.time()

        # train model at each epoch
        t_loss, t_loss_list, batch_list = train(model=model,
                                                train_loader=train_loader,
                                                optimizer=optimizer,
                                                epoch=e)
        v_loss, valid_score = valid(model=model, valid_loader=valid_loader)

        # apply scheduler
        scheduler.step()

        # count for early stop and save log
        early_stop, is_best = early_stopping(score=v_loss, lower_best=True)
        checkpoint.save_log(batch_list=batch_list,
                            epoch=e,
                            train_loss_list_per_batch=t_loss_list,
                            train_loss_per_epoch=t_loss,
                            valid_loss=v_loss)
        checkpoint.save_checkpoint(model=model,
                                   optimizer=optimizer,
                                   is_best=is_best)

        # get filtered score at each epoch
        if filtering:
            filtered_score = get_score(model=model,
                                       data_loader=filtered_loader)
        else:
            filtered_score = None

        # record training time
        iter_time = time.time() - batch_time
        training_time += iter_time

        # visualization for training process
        vis.print_training(env=model_name,
                           EPOCH=epoch,
                           epoch=e,
                           training_time=training_time,
                           iter_time=iter_time,
                           avg_train_loss=t_loss,
                           valid_loss=v_loss,
                           patience=patience,
                           counter=early_stopping.counter)
        vis.loss_plot(env=model_name, checkpoint=checkpoint)
        if anomaly_ratio > 0:
            vis.score_distribution(env=model_name,
                                   anomaly_label=valid_label,
                                   anomaly_score=valid_score,
                                   filtered_score=filtered_score)
            vis.ROC_curve(env=model_name,
                          anomaly_label=valid_label,
                          anomaly_score=valid_score)

        if early_stop:
            break

    test_dataset = Dataset(data=test_data,
                           filtering=False,
                           y_pred=None,
                           time_step=1)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True)
    test_label = test_dataset.get_label(test_label)

    model = checkpoint.load_model(model)

    anomaly_score = get_score(model=model, data_loader=test_loader)
    if filtering:
        filtered_score = get_score(model=model, data_loader=filtered_loader)
    else:
        filtered_score = None

    checkpoint.save_anomaly_score(anomaly_label=test_label,
                                  anomaly_score=anomaly_score,
                                  filtered_score=filtered_score)
    vis.score_distribution(env=model_name,
                           anomaly_label=test_label,
                           anomaly_score=anomaly_score,
                           filtered_score=filtered_score)
    auroc = vis.ROC_curve(env=model_name,
                          anomaly_label=test_label,
                          anomaly_score=anomaly_score)

    with open('./result.csv', 'a') as f:
        f.write(f'{random_state}, {anomaly_type}, {anomaly_ratio}, '
                f'{iqr_multiplier}, {model_name.split("_")[0]}, {auroc}\n')


if __name__ == "__main__":
    main()
