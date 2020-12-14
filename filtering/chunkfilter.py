import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import RobustScaler as RS
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors._base import EfficiencyWarning
from scipy.spatial.distance import mahalanobis
from scipy.sparse import csr_matrix
from scipy.special import erf

warnings.filterwarnings("ignore", category=EfficiencyWarning)


class Chunk_Filter():
    """
    """
    def __init__(self, data: np.ndarray):
        """
        Class instance to filter multivariate time series subsequence
        for autoencoder based unsupervised multivariate time series anomaly
        detection.
        """
        assert data.ndim == 3, "'data' should be chunk data"
        self.data = data
        self.len = self.data.shape[0]

    def fit(self, matrix_type: str = 'nng',
            n_neighbors: int = 20, iqr_multiplier: float = 3,
            normalization: bool = True):
        """
        Method to fit algorithm for filtering.
        """
        assert matrix_type in ['nng', 'full'], \
            "'matrix_type' should be 'nng' for nearest neighbor graph or" \
            "'full' for full distance matrix"

        self.matrix_type = matrix_type
        self.n_neighbors = n_neighbors
        self.normalization = normalization
        self.iqr_multiplier = iqr_multiplier

        self.lof_params = {
            'metric': 'precomputed',
            'n_neighbors': n_neighbors,
            'novelty': False}

        # calculate nng or full distance matrix with dissimilarity measure
        self._get_component(n_neighbors)
        self.distance_matrix = self._get_distance_matrix()

        # fit the model and predict with precomputed distance matrix
        self.model = LocalOutlierFactor(**self.lof_params)
        self.model.fit(self.distance_matrix)
        self.lof_score = -self.model.negative_outlier_factor_
        self.sub_pred = self._subsequence_predict()

        # from subsequence prediction to instance prediction
        self.y_pred = np.zeros((*self.data.shape[:2], 1))
        self.y_pred[self.sub_pred == 1] = 1
        self.y_pred = self.y_pred.reshape(-1)

        return self.y_pred

    def get_metric(self, y_true: np.ndarray):
        """
        Method to estimate model.
        Return accuracy, recall, precision and f1 score.
        """
        assert y_true.ndim == 1, "'y_true' should be 1d array"
        assert y_true.shape == self.y_pred.shape,\
            "the length of 'y_true' is invalid"

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, self.y_pred).ravel()
            accuracy = (tp+tn) / (tn+fp+fn+tp)
            recall = tp / (tp+fn)
            precision = tp / (tp+fp)
            f1 = 2 * (precision * recall) / (precision + recall)
        except ValueError:
            accuracy, recall, precision, f1 = 1, None, None, None

        return accuracy, recall, precision, f1

    def _get_iter(self, i: int, n_neighbor: int):
        if self.matrix_type == 'full':
            return range(i+1, self.len)
        else:
            return range(i+1, min(i + n_neighbor + 2, self.len))

    def _get_component(self, n_neighbor):
        mu_list, e_vector_list, p_list = [], [], []
        for XA in self.data:
            mu, e_vector, p = _calculate_component(XA)
            mu_list.append(mu)
            e_vector_list.append(e_vector)
            p_list.append(p)
        self.dc_array = np.zeros((self.len, self.len))
        self.rc_array = np.zeros((self.len, self.len))
        self.vc_array = np.zeros((self.len, self.len))
        for i in range(self.len-1):
            for j in self._get_iter(i, n_neighbor):
                dc, rc, vc = _dissimilarity(mu_XA=mu_list[i],
                                            mu_XB=mu_list[j],
                                            e_vector_XA=e_vector_list[i],
                                            e_vector_XB=e_vector_list[j],
                                            p_XA=p_list[i],
                                            p_XB=p_list[j],
                                            metric=self.metric)
                self.dc_array[i, j] = dc
                self.rc_array[i, j] = rc
                self.vc_array[i, j] = vc

    def _get_distance_matrix(self):
        dc = self._array_preprocessing(self.dc_array)
        rc = self._array_preprocessing(self.rc_array)
        vc = self._array_preprocessing(self.vc_array)

        i_lower = np.tril_indices(self.len, -1)
        distance_matrix = (dc * rc * vc).reshape(self.len, self.len)
        distance_matrix[i_lower] = distance_matrix.T[i_lower]

        if self.matrix_type == 'nng':
            return csr_matrix(distance_matrix)
        else:
            return distance_matrix

    def _normalization(self, array):
        a = array[array != 0].reshape(-1, 1)
        array[array != 0] = MMS().fit_transform(a).reshape(-1) + 1e-3
        return array

    def _subsequence_predict(self):
        q75, q25 = np.percentile(self.lof_score, [75, 25])
        iqr = q75 - q25
        return self.lof_score > q75 + self.iqr_multiplier*iqr

    def _array_preprocessing(self, array: np.ndarray):
        array = array.copy().reshape(-1)
        if self.normalization:
            array = self._normalization(array)
        return array


def _calculate_component(XA: np.ndarray):
    """
    Calculate each component to calculate dissimilarity for subsequences
    """
    # distance component
    mu_XA = XA.mean(axis=0)

    # rotation component
    pca_XA = PCA()
    pca_XA.fit(SS().fit_transform(XA))
    e_vector_XA = pca_XA.components_

    # variance component
    p_XA = pca_XA.explained_variance_

    return mu_XA, e_vector_XA, p_XA


def _dissimilarity(mu_XA: np.ndarray, mu_XB: np.ndarray,
                   e_vector_XA: np.ndarray, e_vector_XB: np.ndarray,
                   p_XA: np.ndarray, p_XB: np.ndarray):
    """
    Calculate dissimilarity between two subsequences.
    """
    # distance component
    d_c = np.linalg.norm(mu_XA-mu_XB)

    # rotation component
    r_c = np.trace(np.arccos(np.abs(np.dot(e_vector_XA.T, e_vector_XB))))

    # variance component
    v_c = (_kl_divergence(p_XA, p_XB) + _kl_divergence(p_XB, p_XA))/2

    return d_c, r_c, v_c


def _kl_divergence(p: np.ndarray, q: np.ndarray):
    """
    Kullback-Leibler Divergence between two discrete probability distributions
    """
    return np.sum(np.where(p != 0, p * np.log(p/q), 0))
