
import numpy as np

from lieroy.parallel import se3_log
from recova.clustering import compute_distribution


class CovarianceComputationAlgorithm:
    def compute(self, registration_pair):
        raise NotImplementedError('CovarianceComputationAlgorithms must implement compute method.')


class SamplingCovarianceComputationAlgorithm:
    def __init__(self, clustering_algorithm):
        self.clustering_algorithm = clustering_algorithm


    def compute(self, registration_pair):
        results = registration_pair.lie_matrix_of_results()
        clustering = self.clustering_of_pair(registration_pair)
        distribution = compute_distribution(registration_pair.registration_dict(), clustering)

        return np.array(distribution['covariance_of_central'])


    def clustering_of_pair(self, registration_pair):
        clustering = registration_pair.cache[self.clustering_algorithm.__repr__()]

        if not clustering:
            results = registration_pair.lie_matrix_of_results()
            clustering = self.clustering_algorithm.cluster(results, seed=se3_log(registration_pair.ground_truth()))
            registration_pair.cache[self.clustering_algorithm.__repr__()] = clustering

        return clustering




class CensiCovarianceComputationAlgorithm:
    def compute(self, registration_pair):
        pass
