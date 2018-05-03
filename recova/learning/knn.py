
from recova.learning.model import CovarianceEstimationModel

class KnnCovarianceEstimationModel(CovarianceEstimationModel):
    def __init__(self, k=12):
        self.default_k = k

    def fit(self, xs, ys):
        self.kdtree = KDTree(xs)
        self.examples = ys

    def predict(self, xs, p_k=None):
        k = self.default_k if p_k is None else p_k

        distances, indices = self.kdtree.query(xs, k=k)
        predictions = np.zeros((len(xs), 6, 6))

        for i in range(len(xs)):
            exp_dists = np.exp(-distances[i])
            sum_dists = np.sum(exp_dists)
            ratios = exp_dists / sum_dists

            for j in range(len(indices)):
                predicted = np.sum(self.examples[indices[i]] * ratios.reshape(k,1,1), axis=0)
                predictions[i] = predicted

        return predictions

