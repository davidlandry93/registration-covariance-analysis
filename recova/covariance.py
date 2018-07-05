
import argparse
import numpy as np


from lieroy.parallel import se3_log
from recov.registration_algorithm import IcpAlgorithm
from recov.censi import censi_estimate_from_points
from recova.clustering import compute_distribution, IdentityClusteringAlgorithm


class CovarianceComputationAlgorithm:
    def compute(self, registration_pair):
        raise NotImplementedError('CovarianceComputationAlgorithms must implement compute method.')


class SamplingCovarianceComputationAlgorithm:
    def __init__(self, clustering_algorithm=IdentityClusteringAlgorithm()):
        self.clustering_algorithm = clustering_algorithm

    def __repr__(self):
        return 'sampling_covariance_{}'.format(str(self.clustering_algorithm))


    def compute(self, registration_pair):
        def generate_covariance():
            results = registration_pair.lie_matrix_of_results()
            clustering = self.clustering_of_pair(registration_pair)
            distribution = compute_distribution(registration_pair.registration_dict(), clustering)

            if clustering['outlier_ratio'] >= 1.0:
                raise RuntimeError('Empty clustering when running SamplingCovarianceComputationAlgorithm')

            return np.array(distribution['covariance_of_central'])

        return registration_pair.cache.get_or_generate(repr(self) + '_covariance', generate_covariance)



    def clustering_of_pair(self, registration_pair):
        def generate_clustering():
            results = registration_pair.lie_matrix_of_results()
            return self.clustering_algorithm.cluster(results, seed=se3_log(registration_pair.ground_truth()))

        return registration_pair.cache.get_or_generate(repr(self), generate_clustering)




class CensiCovarianceComputationAlgorithm:
    def __init__(self, registration_algorithm = IcpAlgorithm()):
        self.algo = registration_algorithm

    def __repr__(self):
        return 'censi'


    def compute(self, registration_pair):
        def generate_covariance():
            reading = registration_pair.points_of_reading()
            reference = registration_pair.points_of_reference()

            covariance = censi_estimate_from_points(reading, reference, registration_pair.ground_truth(), self.algo)
            return covariance

        return self.cache.get_or_generate(repr(self), generate_covariance)


def covariance_algorithm_factory(algo):
    dict_of_algos = {
        'censi': CensiCovarianceComputationAlgorithm(),
        'sampling': SamplingCovarianceComputationAlgorithm()
    }

    return dict_of_algos[algo]
