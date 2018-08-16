
import argparse
import numpy as np


from lieroy.parallel import se3_log, se3_gaussian_distribution_of_sample
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
            clustering = self.clustering_algorithm.compute(registration_pair)

            group_results = registration_pair.registration_results()[clustering]
            mean, covariance = se3_gaussian_distribution_of_sample(group_results)

            print('Distance from gt: {}'.format(np.linalg.norm(se3_log(np.linalg.inv(registration_pair.ground_truth()) @ mean))))

            return covariance

        return registration_pair.cache.get_or_generate(repr(self) + '_covariance', generate_covariance)




class CensiCovarianceComputationAlgorithm:
    def __init__(self, registration_algorithm = IcpAlgorithm(), sensor_noise_std=0.01):
        self.algo = registration_algorithm
        self.sensor_noise_std = sensor_noise_std

    def __repr__(self):
        return 'censi_{:.4f}'.format(self.sensor_noise_std)


    def compute(self, registration_pair):
        def generate_covariance():
            reading = registration_pair.points_of_reading()
            reference = registration_pair.points_of_reference()

            covariance = censi_estimate_from_points(reading, reference, registration_pair.ground_truth(), self.algo, sensor_noise_std=self.sensor_noise_std)
            return covariance

        return registration_pair.cache.get_or_generate(repr(self), generate_covariance)


def covariance_algorithm_factory(algo):
    dict_of_algos = {
        'censi': CensiCovarianceComputationAlgorithm(),
        'sampling': SamplingCovarianceComputationAlgorithm()
    }

    return dict_of_algos[algo]
