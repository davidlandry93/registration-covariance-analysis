
import argparse
import numpy as np


from lieroy.parallel import se3_log, se3_gaussian_distribution_of_sample
from recov.registration_algorithm import IcpAlgorithm
from recov.censi import censi_estimate_from_points
from recova.clustering import compute_distribution, IdentityClusteringAlgorithm, RegistrationPairClusteringAdapter


class DistributionComputationAlgorithm:
    """
    Compute the mean and covariance the registration results of a pair.
    """
    def compute(self, registration_pair):
        raise NotImplementedError('DistributionComputationAlgorithm must implement compute method.')

    def __repr__(self):
        raise NotImplementedError('DistributionComputationAlgorithm must implement __repr__ method')


class SamplingDistributionComputationAlgorithm(DistributionComputationAlgorithm):
    def __init__(self, clustering_algorithm=RegistrationPairClusteringAdapter(IdentityClusteringAlgorithm())):
        self.clustering_algo = clustering_algorithm

    def __repr__(self):
        return 'sampling_distribution_{}'.format(repr(self.clustering_algo))

    def compute(self, registration_pair):
        def compute_distribution():
            clustering = self.clustering_algo.compute(registration_pair)

            group_results = registration_pair.registration_results()[clustering]
            mean, covariance = se3_gaussian_distribution_of_sample(group_results)

            # Here we return a json instead of numpy arrays to make sure the result can be cached properly.
            # As of 2018-08-17 file_cache.py does not handle embedded numpy arrays because they are not json serializable.
            return {
                'mean': mean.tolist(),
                'covariance': covariance.tolist()
            }

        return registration_pair.cache.get_or_generate(repr(self), compute_distribution)


class FixedCenterSamplingDistributionAlgorithm(DistributionComputationAlgorithm):
    def __init__(self, clustering_algorithm=IdentityClusteringAlgorithm()):
        self.clustering_algo = clustering_algorithm

    def __repr__(self):
        return 'fixed_center_sampling_{}'.format(self.clustering_algo)

    def compute(self, pair):
        clustering = self.clustering_algo.compute(pair)
        group_registration_ts = pair.registration_results()[clustering]

        gt_inv = np.linalg.inv(pair.ground_truth())

        deltas = np.array([se3_log(gt_inv @ t) for t in group_registration_ts])
        cov = (deltas.T @ deltas) / max(len(deltas), 1.0)

        return {
            'mean': pair.ground_truth().tolist(),
            'covariance': cov.tolist()
        }



class CovarianceComputationAlgorithm:
    def compute(self, registration_pair):
        raise NotImplementedError('CovarianceComputationAlgorithms must implement compute method.')


class DistributionAlgorithmToCovarianceAlgorithm(CovarianceComputationAlgorithm):
    def __init__(self, distribution_algo):
        self.distribution_algo = distribution_algo

    def __repr__(self):
        return '{}_covariance'.format(repr(self.distribution_algo))

    def compute(self, registration_pair):
        return np.array(self.distribution_algo.compute(registration_pair)['covariance'])


class SamplingCovarianceComputationAlgorithm:
    def __init__(self, clustering_algorithm=IdentityClusteringAlgorithm()):
        self.distribution_algo = SamplingDistributionComputationAlgorithm(clustering_algorithm)

    def __repr__(self):
        return 'sampling_covariance_{}'.format(repr(self.distribution_algo))

    def compute(self, registration_pair):
        distribution = self.distribution_algo.compute(registration_pair)
        return np.array(distribution['covariance'])




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
