


#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import seaborn as sb
import sys
import torch

from recova.learning.learning import model_from_file
from recova.registration_dataset import registrations_of_dataset
from recova.util import eprint

from lieroy.parallel import se3_gaussian_distribution_of_sample

def covariance_of_central_cluster(dataset, clustering):
    """
    :arg dataset: A full registration dataset.
    :arg clustering: A 2d array describing a clustering.
    """
    central_cluster = find_central_cluster(dataset, clustering)

    registrations = registrations_of_dataset(new_dataset)
    registrations = registration[central_cluster]

    if len(registrations) != 0:
        mean, covariance = se3_gaussian_distribution_of_sample(registrations)
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

def plot_loss_comparison(ax, learning_runs, std=False, labels=[]):
    if len(labels) == len(learning_runs):
        labels_to_use = labels
    else:
        labels_to_use = ['Validation loss, Alpha: {}, Lr: {}'.format(x['metadata']['alpha'], x['metadata']['learning_rate']) for x in learning_runs]


    for i, learning_run in enumerate(learning_runs):
        validation_loss = np.array(learning_run['validation_loss'])
        validation_std = np.array(learning_run['validation_std'])

        epochs = [x * learning_run['metadata']['logging_rate'] for x in range(len(validation_std))]

        ax.plot(epochs, validation_loss, label=labels_to_use[i])
        if std:
            ax.fill_between(range(0, len(validation_loss)), validation_loss + validation_std, validation_loss - validation_std, alpha=0.5)

def plot_single_loss(ax, learning_run, median=False, std=False):
    if median:
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

    epochs = [x * learning_run['metadata']['logging_rate'] for x in range(len(validation_loss))]

    eprint(epochs)

    std_ax = ax.twinx()

    if std:
        ax.fill_between(epochs, validation_data[2], validation_data[0], alpha=0.5)

    ax.plot(epochs, validation_data[1], label='Validation loss')
    # std_ax.plot(epochs, validation_std, label='Validation standard deviation', linestyle='--', alpha=0.7)

    if std:
        ax.fill_between(epochs, optimization_data[2], optimization_data[0], alpha=0.5)

    ax.plot(epochs, optimization_data[1], label='Optimization loss')
    # std_ax.plot(epochs, optimization_std, label='Optimization standard deviation', linestyle='--', alpha=0.7)



def plot_loss_on_time(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--median', action='store_true', help='Wether we should use medians or averages to plot the loss.')
    parser.add_argument('-std', action='store_true', help='Show the standard deviations/interquartile spread.')
    parser.add_argument('--labels', '-l', type=str, default='', help='Comma separated list of labels for the different plots.')
    parser.add_argument('-t', '--title', type=str, default='Evolution of validation loss during learning')
    args = parser.parse_args(args)

    learning_run = json.load(sys.stdin)

    fig, ax = plt.subplots()

    if isinstance(learning_run, dict):
        plot_single_loss(ax, learning_run, median=args.median, std=args.std, )
    elif isinstance(learning_run, list):
        plot_loss_comparison(ax, learning_run, std=args.std, labels=args.labels.split(','))

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frobenius norm  error')
    ax.set_title(args.title)
    plt.show()


def generate_axis_configuration(dataset, indices_of_axis):
    location_count = dict()
    for i in indices_of_axis:
        location = dataset['data']['pairs'][i]['dataset']

        if location in location_count:
            location_count[location] += 1
        else:
            location_count[location] = 1

    major_ticks = [0]
    for location in sorted(location_count):
        major_ticks.append(major_ticks[-1] + location_count[location])

    minor_ticks = []
    minor_ticks_labels = []
    for i in range(len(major_ticks) - 1):
        minor_ticks.append(float(major_ticks[i+1] - major_ticks[i]) / 2. + major_ticks[i])
        minor_ticks_labels.append(sorted(location_count.keys())[i])

    return major_ticks, minor_ticks, minor_ticks_labels



def plot_activation_matrix(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The learning dataset used to generate the model')
    parser.add_argument('learningrun', help='The learning run to plot')
    parser.add_argument('model', help='The CELLO model to plot')
    args = parser.parse_args(args)

    model = model_from_file(args.model, 'cello')

    with open(args.dataset) as f:
        dataset = json.load(f)

    with open(args.learningrun) as f:
        learning_run = json.load(f)

    xs = np.array(dataset['data']['xs'])
    ys = np.array(dataset['data']['ys'])

    learning_indices = np.array(learning_run['train_set'])
    validation_indices = np.array(sorted(learning_run['validation_set']))
    sort_of_learning_examples = np.argsort(learning_indices)

    # Generate the activation data
    activation_matrix = np.zeros((len(learning_indices), len(validation_indices)))
    for i, q in enumerate(validation_indices):
        distances = model.compute_distances(xs[q])
        weights = model.distances_to_weights(torch.Tensor(distances))
        activation_matrix[:, i] = weights[sort_of_learning_examples]


    fig, ax = plt.subplots()
    sb.heatmap(activation_matrix, ax=ax, square=True)


    # Compute the data to configure the x axis labels
    major_ticks, minor_ticks, labels = generate_axis_configuration(dataset, validation_indices)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels('')
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(labels, minor=True, rotation=90)

    # Compute the data to configure the y axis labels
    major_ticks, minor_ticks, labels = generate_axis_configuration(dataset, learning_indices)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels('')
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticklabels(labels, minor=True)

    ax.tick_params(axis='both', which='minor', length=0)

    ax.set_xlabel('Validation pairs')
    ax.set_ylabel('Training pairs')
    ax.set_title('Weight of learning examples when predicting validation examples')

    fig.set_size_inches(10, 20)

    # plt.tight_layout()
    plt.show()


functions_of_plots = {
    'cov_on_density': plot_cov_against_density,
    'loss_on_time': plot_loss_on_time,
    'activation_matrix': plot_activation_matrix,
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
