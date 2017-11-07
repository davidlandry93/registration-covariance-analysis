


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
    """
    :arg dataset: A full registration dataset.
    :arg clustering: A 2d array describing a clustering.
    """
    new_dataset = dataset.copy()
    central_cluster = find_central_cluster(new_dataset, clustering)
    new_dataset = filter_with_cluster(new_dataset, central_cluster)

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
    clusterings = json.load(sys.stdin)

    xs = []
    ys = []
    sizes = []
    for i, clustering in enumerate(clusterings['data']):
        xs.append(clustering['density'])
        sizes.append(len(clustering['central_cluster']))

        covariance = np.array(clustering['covariance_of_central'])
        ys.append(np.trace(covariance))

    fig, ax1 = plt.subplots()
    plot1 = ax1.plot(xs, ys, linestyle='-', marker='o', label='Trace of covariance matrix', color='black')
    ax1.set_xlabel('Density Gain')
    ax1.set_ylabel('Trace of covariance matrix')

    ax2 = ax1.twinx()
    plot2 = ax2.plot(xs, sizes, label='N of points in cluster', linestyle='--', marker='s', color='0.5')
    ax2.set_xlabel('Density Gain')
    ax2.set_ylabel('N of points in cluster')

    plots = plot1 + plot2
    labels = [x.get_label() for x in plots]
    ax1.legend(plots, labels, loc=0)


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
