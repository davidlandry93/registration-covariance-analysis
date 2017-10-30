

#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import sys

from recova.covariance_of_registrations import distribution_of_registrations
from recova.find_center_cluster import find_central_cluster, filter_with_cluster
from recova.registration_dataset import registrations_of_dataset
from recova.util import eprint


def covariance_of_central_cluster(dataset, clustering):
    new_dataset = dataset.copy()
    clustering_data = clustering['data']
    new_dataset = filter_with_cluster(new_dataset, find_central_cluster(new_dataset, clustering_data))

    registrations = registrations_of_dataset(new_dataset)

    if len(registrations) != 0:
        mean, covariance = distribution_of_registrations(registrations)
    else:
        return np.zeros((6,6))

    return covariance


def plot_cov_against_density(args):
    """
    Plot the evolution of the covariance when we change the clustering radius.
    """
    parser = argparse.ArgumentParser(prog='cov_on_density')
    parser.add_argument('dataset', type=str, help='The dataset that was clustered')
    args = parser.parse_args(args)

    clusterings = json.load(sys.stdin)
    with open(args.dataset) as dataset_file:
        dataset = json.load(dataset_file)

    registrations = registrations_of_dataset(dataset)

    covariances = np.empty((len(clusterings), 6, 6))
    for i, clustering in enumerate(clusterings):
        print(i)
        covariance = covariance_of_central_cluster(dataset, clustering)
        covariances[i] = covariance
        print('{} yield a trace of {}'.format(clustering['metadata']['radius'], np.trace(covariance)))
        print(covariance)

    xs = []
    ys = []
    for i, covariance in enumerate(covariances):
        xs.append(clusterings[i]['metadata']['radius'])
        ys.append(np.trace(covariance))

    plt.plot(xs, ys, linestyle='-', marker='o')
    plt.show()

functions_of_plots = {
    'cov_on_density': plot_cov_against_density
}


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Name of the plot to show')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.name not in functions_of_plots:
        raise RuntimeError('Plot \"{}\" is undefined.'.format(args.name))


    functions_of_plots[args.name](args.rest)

if __name__ == '__main__':
    cli()
