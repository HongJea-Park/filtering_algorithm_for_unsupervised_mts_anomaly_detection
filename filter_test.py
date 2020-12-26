import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
from sklearn.preprocessing import MinMaxScaler

from data.synthetic import Generator
from filtering.chunkfilter import Chunk_Filter
from utils.seq2chunk import seq2chunk


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def main(random_state, n_neighbors, filter_size, anomaly_ratio, anomaly_type,
         matrix_type, iqr_multiplier):
    # about synthetic data
    data_len = 300000
    num_feature = 25
    n_neighbors = 20
    filter_size = 30
    normalization = True
    t_idx = int(data_len * 0.3)
    v_idx = int(data_len * 0.5)
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

    start = time.time()
    mts_filter = Chunk_Filter(data=chunk_data)
    filter_pred = mts_filter.fit(
        matrix_type=matrix_type,
        n_neighbors=n_neighbors,
        iqr_multiplier=iqr_multiplier,
        normalization=normalization)
    acc, recall, precision, f1 = \
        mts_filter.get_metric(train_valid_label)
    filter_result = {
        'random_state': random_state,
        'anomaly_type': anomaly_type,
        'anomaly_ratio': anomaly_ratio,
        'matrix_type': matrix_type,
        'iqr_multiplier': iqr_multiplier,
        'accuracy': acc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'time': time.time() - start,
        'filtered': filter_pred.sum() / filter_pred.shape[0]}
    for k, v in filter_result.items():
        print(f'{k}: {v}')
    print('-'*50)
    return filter_result


def result_to_csv(results):
    df = {}
    df['random_state'] = []
    df['anomaly_type'] = []
    df['anomaly_ratio'] = []
    df['matrix_type'] = []
    df['iqr_multiplier'] = []
    df['accuracy'] = []
    df['recall'] = []
    df['precision'] = []
    df['f1'] = []
    df['time'] = []
    df['filtered'] = []

    for result in results:
        df['random_state'].append(result['random_state'])
        df['anomaly_type'].append(result['anomaly_type'])
        df['anomaly_ratio'].append(result['anomaly_ratio'])
        df['matrix_type'].append(result['matrix_type'])
        df['iqr_multiplier'].append(result['iqr_multiplier'])
        df['accuracy'].append(result['accuracy'])
        df['recall'].append(result['recall'])
        df['precision'].append(result['precision'])
        df['f1'].append(result['f1'])
        df['time'].append(result['time'])
        df['filtered'].append(result['filtered'])

    df = pd.DataFrame(df, columns=[k for k in df.keys()])
    df.to_csv(f'filtering_result.csv', index=False)


if __name__ == "__main__":

    n_cpu = mp.cpu_count()
    random_state_list = [42, 21, 7, 14, 28]
    n_neighbors_list = [20]
    filter_size = [30]
    anomaly_ratio_list = [0.001, 0.005, 0.01, 0.05, 0.1]
    anomaly_type_list = ['type1', 'type2']
    matrix_type = ['full', 'nng']
    iqr_multiplier_list = [1.5, 3.0]

    hyper_parameter_list = list(product(*[random_state_list,
                                          n_neighbors_list,
                                          filter_size,
                                          anomaly_ratio_list,
                                          anomaly_type_list,
                                          matrix_type,
                                          iqr_multiplier_list]))

    mp.freeze_support()

    with mp.Pool(processes=n_cpu) as pool:
        results = pool.starmap(main, hyper_parameter_list)

    result_to_csv(results)
