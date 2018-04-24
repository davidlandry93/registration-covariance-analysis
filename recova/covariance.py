
import numpy as np


from lieroy.parallel import se3_log
from recov.registration_algorithm import IcpAlgorithm
from recov.censi import censi_estimate_from_clouds
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
    def __init__(self, registration_algorithm = IcpAlgorithm()):
        self.algo = registration_algorithm


    def compute(self, registration_pair):
        path_to_reading = registration_pair.path_to_reading_pcd()
        path_to_reference = registration_pair.path_to_reference_pcd()

        covariance = censi_estimate_from_clouds(path_to_reading, path_to_reference, registration_pair.ground_truth(), self.algo)

        return np.array(covariance)
