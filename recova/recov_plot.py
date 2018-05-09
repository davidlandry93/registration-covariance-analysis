


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


def plot_loss_on_time(args):
    learning_run = json.load(sys.stdin)

    fig, ax = plt.subplots()

    loss = np.array(learning_run['validation_loss'])
    optimization_loss = np.array(learning_run['optimization_loss'])
    loss_std = np.array(learning_run['validation_std'])

    validation_errors = np.array(learning_run['validation_errors'])
    validation_percentiles = np.percentile(validation_errors, [0.0, 0.25, 0.5, 0.75, 100], axis=1)

    optimization_errors = np.array(learning_run['optimization_errors'])
    optimization_percentiles = np.percentile(optimization_errors, [0.0, 0.25, 0.5, 0.75, 100], axis=1)

    print('Last validation errors')
    print(np.sort(validation_errors[-1]))
    print(validation_percentiles[4])

    # ax.fill_between(range(0, len(loss)), validation_percentiles[4], validation_percentiles[0], alpha=0.3, color='blue')
    ax.fill_between(range(0, len(loss)), validation_percentiles[3], validation_percentiles[1], alpha=0.5)
    ax.plot(validation_percentiles[2], label='Validation loss')

    # ax.fill_between(range(0, len(loss)), optimization_percentiles[4], optimization_percentiles[0], alpha=0.3, color='orange')
    # ax.fill_between(range(0, len(loss)), optimization_percentiles[4], optimization_percentiles[0], alpha=0.5)
    # ax.plot(optimization_percentiles[2], label='Optimization loss')

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frobenius norm  error (median)')
    ax.set_title('Evolution of validation loss during learning')

    plt.show()


functions_of_plots = {
    'cov_on_density': plot_cov_against_density,
    'loss_on_time': plot_loss_on_time
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
