


#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import sys

from recova.covariance_of_registrations import distribution_of_registrations
from recova.find_center_cluster import find_central_cluster 
from recova.registration_dataset import registrations_of_dataset
from recova.util import eprint


def covariance_of_central_cluster(dataset, clustering):
    """
    :arg dataset: A full registration dataset.
    :arg clustering: A 2d array describing a clustering.
    """
    central_cluster = find_central_cluster(dataset, clustering)

    registrations = registrations_of_dataset(new_dataset)
    registrations = registration[central_cluster]

    if len(registrations) != 0:
        mean, covariance = distribution_of_registrations(registrations)
    else:
        return np.zeros((6,6))

    return covariance


def plot_clustering_series(covariance_trace_ax, n_points_ax, clusterings):
    xs = []
    ys = []
    sizes = []
    for i, clustering in enumerate(clusterings['data']):
        xs.append(clustering['radius'])
        sizes.append(1.0 - float(clustering['outlier_ratio']))

        covariance = np.array(clustering['covariance_of_central'])
        ys.append(np.trace(covariance))

    print(clusterings['metadata'])
    plot1 = covariance_trace_ax.plot(xs, ys, linestyle='-', marker='o', label='{} ({})'.format(clusterings['metadata']['dataset'], clusterings['metadata']['reference']))
    plot2 = n_points_ax.plot(xs, sizes, label='N of points in cluster', linestyle='--', marker='s')
    covariance_trace_ax.set_ylabel('Trace of covariance matrix')


def plot_cov_against_density(args):
    """
    Plot the evolution of the covariance when we change the clustering radius.
    """
    clusterings = json.load(sys.stdin)

    fig, ax = plt.subplots()
    n_points_ax = ax.twinx()

    if isinstance(clusterings, dict):
        plot_clustering_series(ax, n_points_ax, clusterings)
    elif isinstance(clusterings, list):
        for series in clusterings:
            plot_clustering_series(ax, n_points_ax, series)

    ax.legend()
    ax.set_xlabel('Clustering radius')
    ax.set_ylabel('Proportion of inliers')
    ax.set_title('Trace of covariance matrix according to clustering radius')

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
