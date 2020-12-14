import numpy as np


class Generator():
    def __init__(self, random_state: int = 42):
        """
        Class object to generate syntehtic data.
        """
        self.random_state = random_state
        self.pattern = {0: np.sin, 1: np.cos}

    def generate(self, shape: tuple = (300000, 25),
                 anomaly_ratio: float = 0.05,
                 split_ratio: list = [0.3, 0.2, 0.5],
                 min_feature_ratio: float = 0.2,
                 max_feature_ratio: float = 0.5,
                 min_len: int = 50, max_len: int = 200,
                 anomaly_type: str = 'type1'):
        """
        Generate multivariate time series synthetic data.
        """
        assert anomaly_type in ['type1', 'type2'], \
            "Wrong anomaly type. Choose between 'type1' and 'type2'."
        assert sum(split_ratio) == 1, \
            "The sum of ratio of train, valid and test set should be 1."
        assert max_len >= min_len, "'max_len' should be greater than 'min_len"

        # The length of sequence and the number of feature
        self.len, self.num_feature = shape
        self.min_len = min_len
        self.max_len = max_len
        self.anomaly_type = anomaly_type
        self.data_len_list = [int(self.len*s) for s in np.cumsum(split_ratio)]
        self.idx1 = self.data_len_list[0]
        self.idx2 = self.data_len_list[1]
        self.ano_set = []
        self.ano_features = []

        # Fix the values to generate sequence for each feature
        np.random.seed(self.random_state)
        self.seed = np.random.choice([0, 1], size=self.num_feature)
        self.t = np.arange(0, self.len, dtype=int)
        self.t0 = np.random.choice(np.arange(50, 100), size=self.num_feature)
        self.w = np.random.choice(np.arange(20, 40), size=self.num_feature)
        self.noise = np.random.normal(size=(self.len, self.num_feature))
        self.s_factor = np.random.uniform(3, 5, size=self.num_feature)
        self.n_factor = np.random.uniform(0.1, 0.3, size=self.num_feature)
        self.intercept = np.random.uniform(-5, 5, size=self.num_feature)

        # Concatenate each sequence
        self.data = {}
        for f in range(self.num_feature):
            cycle = (self.t-self.t0[f]) / self.w[f]
            self.data[f] = self.s_factor[f] * self.pattern[self.seed[f]](cycle)
            self.data[f] += self.n_factor[f] * self.noise[:, f]
            self.data[f] += self.intercept[f]
        self.data = np.array(list(self.data.values())).T
        self.label = np.zeros(shape=self.len)

        # Select the subsets of features
        self.min_feature = int(self.num_feature*min_feature_ratio)
        self.max_feature = int(self.num_feature*max_feature_ratio)
        self.feature_list = np.random.choice(a=np.arange(self.num_feature),
                                             size=self.max_feature,
                                             replace=False)
        self.fixed_features = np.random.choice(a=self.feature_list,
                                               size=self.min_feature,
                                               replace=False)
        self.random_features = np.setdiff1d(
            self.feature_list, self.fixed_features)

        # Sampling anomaly index to test set and inject subsequence anomaly
        self._inject_anomaly(self.idx2, self.len, 0.05)
        if anomaly_ratio > 0:
            self._inject_anomaly(0, self.idx1, anomaly_ratio)
            self._inject_anomaly(self.idx1, self.idx2, anomaly_ratio)

    def _inject_anomaly(self, s_idx: int, e_idx: int, anomaly_ratio: float):
        """
        Inject subsequence anomaly in generated dataset.
        """
        # Sampling the number of anomaly subsequences and length
        num_ano = int((e_idx-s_idx) * anomaly_ratio)
        len_list = []
        total_len = num_ano
        while total_len > 0:
            len_ = self._sampling_ano_len(total_len)
            if len_:
                len_list.append(len_)
                total_len -= len_
            else:
                total_len = 0

        # Sampling the anomaly feature lists for anomaly subsequences
        n_feature = np.random.choice(
            a=np.arange(self.max_feature-self.min_feature),
            size=len(len_list),
            replace=True)
        ano_features = [self._select_feature(f) for f in n_feature.tolist()]

        # Sampling anomaly data raw index
        ano_set = []
        for len_ in len_list:
            self._sampling_ano_idx(ano_set, s_idx, e_idx, len_)

        # Inject anomaly subsequences
        for idx, features in zip(ano_set, ano_features):
            noise = np.random.normal(size=(len(idx), len(features)))
            if self.anomaly_type == 'type1':
                self._anomaly_type1(idx, features, noise)
            else:
                self._anomaly_type2(idx, features, noise)

        self.ano_set.extend(ano_set)
        self.ano_features.extend(ano_features)

    def _sampling_ano_idx(self, ano_set: list, s_idx: int, e_idx: int,
                          len_: int):
        while True:
            t = np.random.choice(a=self.t[s_idx:e_idx])
            if self.label[t:t+len_].sum() > 0:
                continue
            ano_set.append(np.arange(t, t+len_))
            self.label[t:t+len_] = 1
            break

    def _sampling_ano_len(self, total_len: int):
        if total_len > self.max_len:
            return np.random.choice(np.arange(self.min_len, self.max_len))
        elif total_len > self.min_len:
            return total_len
        else:
            return 0

    def _select_feature(self, num_feature: int):
        selected = np.random.choice(a=self.random_features,
                                    size=num_feature,
                                    replace=False)
        selected = np.append(self.fixed_features, selected)
        return selected

    def _anomaly_type1(self, idx: np.ndarray, features: np.ndarray,
                       noise: np.ndarray):
        # if len(idx) > min(self.w)*3:
        if False:
            m = (self.data[idx[-1], features])
            s = self.data[idx[-1], features] - self.data[idx[0], features]
            m -= s*0.2
            slope = self.data[idx[-1], features]-m
            slope /= (idx[-1] - idx[0])
            intercept = m
            new_y = (np.arange(idx[-1]-idx[0]+1).reshape(-1, 1)*slope)
            idx = idx.reshape(-1, 1)
            features = features.reshape(1, -1)
            self.data[idx, features] = new_y + intercept
            self.data[idx, features] += self.n_factor[features]*noise
        else:
            m = (self.data[idx[-1], features])
            s = self.data[idx[-1], features] - self.data[idx[0], features]
            m -= s*0.2
            idx = idx.reshape(-1, 1)
            features = features.reshape(1, -1)
            self.data[idx, features] = self.n_factor[features]*noise + m

    def _anomaly_type2(self, idx: np.ndarray, features: np.ndarray,
                       noise: np.ndarray):
        t = np.tile(idx, reps=(len(features), 1)).T
        c = (t-self.t0[features]) / self.w[features]
        # n_f = np.random.uniform(5, 10, size=features.shape)
        n_f = 5
        for i, f in enumerate(features):
            c[:, i] = \
                self.s_factor[f]*self.pattern[self.seed[f]](c[:, i])
        idx = idx.reshape(-1, 1)
        features = features.reshape(1, -1)
        self.data[idx, features] = c+self.n_factor[features]*n_f*noise
        self.data[idx, features] += self.intercept[features]
