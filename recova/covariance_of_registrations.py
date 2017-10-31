#!/usr/bin/python3

import argparse
import json
import multiprocessing
import numpy as np
import sys

from recova.registration_dataset import registrations_of_dataset
from pylie import se3_log, se3_exp


def compute_one_perturbation(inv_of_mean, registration):
    perturbation_matrix = np.dot(inv_of_mean, registration)
    perturbation_lie = se3_log(perturbation_matrix)

    covariance = np.dot(perturbation_lie.reshape((6,1)), perturbation_lie.reshape((1,6)))

    return (perturbation_lie, covariance)


def distribution_of_registrations(registrations):
    """
    :arg registrations: An iterable of 4x4 SE(3) transformations.
    :returns: mean, covariance. The mean and covariance of the distribution of registrations.
    """
    mean = registrations[0]
    inv_of_mean = np.linalg.inv(mean)

    former_perturbation = np.full(6, np.inf)
    average_perturbation = np.zeros(6)
    i = 0
    while np.linalg.norm(former_perturbation - average_perturbation) > 1e-4 and i < 15:
        print('Delta: {}'.format(np.linalg.norm(former_perturbation - average_perturbation)))
        former_perturbation = average_perturbation
        average_perturbation = np.zeros(6)
        covariance = np.zeros((6,6))

        results = []
        for registration in registrations:
            results.append(compute_one_perturbation(inv_of_mean, registration))

        perturbations, covariances = zip(*results)
        perturbations = np.array(perturbations)
        covariances = np.array(covariances)

        average_perturbation = perturbations.sum(axis=0) / len(registrations)
        covariance = covariances.sum(axis=0) / (len(registrations) - 1)

        mean = np.dot(se3_exp(average_perturbation), mean)
        inv_of_mean = np.linalg.inv(mean)
        i += 1

    return mean, covariance


def cli():
    dataset = json.load(sys.stdin)
    registrations = registrations_of_dataset(dataset)

    mean, covariance = distribution_of_registrations(registrations)

    json.dump({'mean': mean.tolist(), 'covariance': covariance.tolist()}, sys.stdout)

if __name__ == '__main__':
    cli()
