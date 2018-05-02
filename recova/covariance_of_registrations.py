#!/usr/bin/python3

import argparse
import json
import multiprocessing
import numpy as np
import sys

from recova.registration_dataset import registrations_of_dataset
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
    dataset = json.load(sys.stdin)
    registrations = registrations_of_dataset(dataset)

    mean, covariance = distribution_of_registrations(registrations)

    json.dump({'mean': mean.tolist(), 'covariance': covariance.tolist()}, sys.stdout)


if __name__ == '__main__':
    cli()
