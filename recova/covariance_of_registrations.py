#!/usr/bin/python3 

import argparse
import json
import numpy as np
import sys

from recova.registration_dataset import registrations_of_dataset
from pylie import se3_log, se3_exp

def distribution_of_registrations(registrations):
    mean = registrations[0]

    for i in range(10):
        average_perturbation = np.zeros(6)
        covariance = np.zeros((6,6))

        for registration in registrations:
            perturbation_matrix = np.dot(np.linalg.inv(mean), registration)
            perturbation_lie = se3_log(perturbation_matrix)

            average_perturbation += perturbation_lie
            covariance += np.dot(perturbation_lie.reshape((6,1)), perturbation_lie.reshape((1,6)))

        average_perturbation /= len(registrations)
        covariance /= len(registrations) - 1

        print('Avg perturbation of iter: ')
        print(se3_exp(average_perturbation))
        mean = np.dot(se3_exp(average_perturbation), mean)

    return mean, covariance


def cli():
    dataset = json.load(sys.stdin)
    registrations = registrations_of_dataset(dataset)

    mean, covariance = distribution_of_registrations(registrations)

    print(mean)
    print(covariance)


if __name__ == '__main__':
    cli()
