#!/usr/bin/python3

import argparse
import json
import multiprocessing
import numpy as np
import sys

from recova.clustering import DensityThresholdClusteringAlgorithm
from recova.covariance import SamplingCovarianceComputationAlgorithm
from recova.registration_dataset import registrations_of_dataset
from recova.registration_result_database import RegistrationPairDatabase
from lieroy.parallel import se3_log, se3_exp, se3_gaussian_distribution_of_sample


def compute_one_perturbation(inv_of_mean, registration):
    perturbation_matrix = np.dot(inv_of_mean, registration)
    perturbation_lie = se3_log(perturbation_matrix)

    covariance = np.dot(perturbation_lie.reshape((6,1)), perturbation_lie.reshape((1,6)))

    return (perturbation_lie, covariance)


def distribution_of_registrations(registrations):
    """
    :arg registrations: A ndarray of 4x4 SE(3) transformations.
    :returns: mean, covariance. The mean and covariance of the distribution of registrations.
    """
    return se3_gaussian_distribution_of_sample(registrations)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str)
    parser.add_argument('location', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('--density-filter', type=float, default=1e3)
    args = parser.parse_args()

    clustering = DensityThresholdClusteringAlgorithm(args.density_filter, k=100)
    covariance_algo = SamplingCovarianceComputationAlgorithm(clustering_algorithm=clustering)

    db = RegistrationPairDatabase(args.database)
    registration_pair = db.get_registration_pair(args.location, args.reading, args.reference)
    covariance = covariance_algo.compute(registration_pair)
    mean = registration_pair.ground_truth()

    json.dump({'mean': mean.tolist(), 'covariance': covariance.tolist()}, sys.stdout)


if __name__ == '__main__':
    cli()
