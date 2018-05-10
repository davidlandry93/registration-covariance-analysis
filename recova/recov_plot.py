


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--median', action='store_true', help='Wether we should use medians or averages to plot the loss.')
    parser.add_argument('-std', action='store_true', help='Show the standard deviations/interquartile spread.')
    args = parser.parse_args(args)

    learning_run = json.load(sys.stdin)


    if args.median:
        validation_errors = np.array(learning_run['validation_errors'])
        validation_data = np.percentile(validation_errors, [0.25, 0.5, 0.75], axis=1)

        optimization_errors = np.array(learning_run['optimization_errors'])
        optimization_data = np.percentile(optimization_errors, [0.25, 0.5, 0.75], axis=1)
    else:
        optimization_loss = np.array(learning_run['optimization_loss'])
        optimization_std = np.array(learning_run['optimization_errors']).std(axis=1)

        optimization_data = np.empty((3, len(optimization_std)))
        optimization_data[0] = optimization_loss - optimization_std
        optimization_data[1] = optimization_loss
        optimization_data[2] = optimization_loss + optimization_std

        validation_loss = np.array(learning_run['validation_loss'])
        validation_std = np.array(learning_run['validation_std'])

        validation_data = np.empty((3, len(learning_run['validation_std'])))
        validation_data[0] = validation_loss - validation_std
        validation_data[1] = validation_loss
        validation_data[2] = validation_loss + validation_std


    fig, ax = plt.subplots()
    std_ax = ax.twinx()

    if args.std:
        ax.fill_between(range(0, validation_data.shape[1]), validation_data[2], validation_data[0], alpha=0.5)

    ax.plot(validation_data[1], label='Validation loss')
    std_ax.plot(validation_std, label='Validation standard deviation', linestyle='--', alpha=0.7)

    if args.std:
        ax.fill_between(range(0, validation_data.shape[1]), optimization_data[2], optimization_data[0], alpha=0.5)

    ax.plot(optimization_data[1], label='Optimization loss')
    std_ax.plot(optimization_std, label='Optimization standard deviation', linestyle='--', alpha=0.7)

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
