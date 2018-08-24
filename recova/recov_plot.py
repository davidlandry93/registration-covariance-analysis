#!/usr/bin/env python3

import argparse
import json
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import seaborn as sns
import sys
import torch

import recov.datasets
import recova.clustering
from recova.descriptor.factory import descriptor_factory
from recova.learning.learning import model_from_file
from recova.registration_dataset import registrations_of_dataset
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint, parallel_starmap_progressbar

from lieroy import se3
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

def plot_single_loss(ax, learning_run, median=False, std=False, kll=False):
    if median:
        validation_errors = np.array(learning_run['validation_errors'])
        validation_data = np.percentile(validation_errors, [0.25, 0.5, 0.75], axis=1)

        optimization_errors = np.array(learning_run['optimization_errors'])
        optimization_data = np.percentile(optimization_errors, [0.25, 0.5, 0.75], axis=1)
    elif kll:
        kll_training = np.arrau(learning_run['kll_validation'])

        kll_errors = np.array(learning_run['kll_errors'])
        optimization_data = np.percentile(kll_errors, [0.25, 0.5, 0.75], axis=1)
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
    parser.add_argument('-kll', action='store_true')
    args = parser.parse_args(args)

    learning_run = json.load(sys.stdin)

    fig, ax = plt.subplots()

    if isinstance(learning_run, dict):
        plot_single_loss(ax, learning_run, median=args.median, std=args.std, kll=args.kll)
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

    xs = torch.Tensor(np.array(dataset['data']['xs']))
    ys = torch.Tensor(np.array(dataset['data']['ys']))

    learning_indices = np.array(sorted(learning_run['train_set']))
    validation_indices = np.array(sorted(learning_run['validation_set']))
    sort_of_learning_examples = np.argsort(learning_indices)

    print(learning_indices)

    # Generate the activation data
    activation_matrix = np.zeros((len(learning_indices), len(validation_indices)))
    for i, q in enumerate(validation_indices):
        distances = model.compute_distances(xs[q])
        weights = model.distances_to_weights(torch.Tensor(distances))
        activation_matrix[:, i] = weights[sort_of_learning_examples]


    fig, ax = plt.subplots()
    sns.heatmap(activation_matrix.T, ax=ax, square=True, cbar_kws={'shrink': 0.5})


    # Compute the data to configure the x axis labels
    major_ticks, minor_ticks, labels = generate_axis_configuration(dataset, validation_indices)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels('')
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticklabels(labels, minor=True)

    # Compute the data to configure the y axis labels
    major_ticks, minor_ticks, labels = generate_axis_configuration(dataset, learning_indices)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels('')
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(labels, minor=True, rotation=90)

    ax.tick_params(axis='both', which='minor', length=0)

    ax.grid(color='white', linestyle='--')

    ax.set_ylabel('Validation pairs')
    ax.set_xlabel('Training pairs')
    ax.set_title('Weight of learning examples when predicting validation examples')

    plt.tight_layout()
    plt.show()




def plot_trajectory_evaluation(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='Path to registration database')
    parser.add_argument('location', type=str, help='Name of the location to work on.')
    parser.add_argument('dataset', type=str, help='Path to point cloud dataset.')
    parser.add_argument('learning_dataset', type=str)
    parser.add_argument('model', type=str, help='Path to covariance prediction model.')
    args = parser.parse_args(args)

    np.set_printoptions(precision=3)

    with open(args.learning_dataset) as f:
        learning_dataset = json.load(f)
        descriptor_algo = descriptor_factory(learning_dataset['metadata']['descriptor_config'])

    dataset = recov.datasets.KittiDataset(pathlib.Path(args.dataset))
    print(dataset.times)
    database = RegistrationPairDatabase(args.database)

    pairs = database.registration_pairs()
    pairs = list(filter(lambda x: x.dataset == args.location, pairs))

    model = model_from_file(args.model, 'cello')
    predictions = predict_covariances(pairs, descriptor_algo, model)

    gt_trajectory = np.empty((len(pairs) + 1, 4, 4))
    gt_trajectory[0] = np.identity(4)
    for i in range(1, dataset.n_clouds()):
        gt_trajectory[i] = gt_trajectory[i - 1] @ dataset.ground_truth(i, i-1)

    cum_covariances = make_cumulative_covariances(gt_trajectory, predictions)

    clustering_algo = recova.clustering.CenteredClusteringAlgorithm(radius=0.1, k=20, n_seed_init=20)
    clustering_algo = recova.clustering.RegistrationPairClusteringAdapter(clustering_algo)
    sampled_trajectory = make_sampled_trajectory(database, args.location, len(gt_trajectory), clustering_algo)

    fig, ax = plt.subplots()
    plot_trajectory_translation(gt_trajectory, ax, palette='Blues')
    plot_trajectory_translation(sampled_trajectory, ax, palette='Oranges')
    # plot_trajectory_rotation(dataset.times, gt_trajectory, ax)
    for i in range(0, len(pairs), 3):
        plot_covariance(gt_trajectory[i], cum_covariances[i], ax)
        plot_covariance(gt_trajectory[i], predictions[i], ax, color='blue')

    plt.show()


def make_cumulative_covariances(trajectory, predictions):
    cum_covariances = np.empty((len(trajectory), 6, 6))
    cum_covariances[0] = np.zeros((6,6))
    for i in range(1, len(trajectory)):
        delta = trajectory[i]
        adjoint = se3.adjoint(delta)
        rotated_covariance = adjoint @ (predictions[i - 1] @ adjoint.T)

        print(delta)
        print(adjoint)
        print(rotated_covariance)
        print('---')

        cum_covariances[i] = cum_covariances[i-1] + rotated_covariance

    return cum_covariances


def make_sampled_trajectory(database, location, trajectory_length, clustering_algo):
    trajectory = np.empty((trajectory_length, 4, 4))
    trajectory[0] = np.identity(4)

    pairs = [database.get_registration_pair(location, i, i-1) for i in range(1, trajectory_length)]
    clusterings = parallel_starmap_progressbar(compute_clustering, [(pair, clustering_algo) for pair in pairs])

    for i in range(1, trajectory_length):
        pair = database.get_registration_pair(location, i, i-1)
        results = pair.registration_results()

        t = random.choice(results[clusterings[i - 1]])
        trajectory[i] = trajectory[i-1] @ t

    return trajectory


def compute_clustering(pair, clustering_algo):
    return clustering_algo.compute(pair)


def predict_covariance(pair, descriptor_algo):
    return descriptor_algo.compute(pair)

def predict_covariances(pairs, descriptor_algo, model):
    descriptors = parallel_starmap_progressbar(predict_covariance, [(pair, descriptor_algo) for pair in pairs])

    descriptors_np = np.empty((len(pairs), len(descriptor_algo.labels())))
    for i, descriptor in enumerate(descriptors):
        descriptors_np[i] = descriptor

    predictions = model.predict(descriptors_np)

    return predictions


def plot_covariance(mean, covariance, ax, color='black'):
    eigvals, eigvecs = np.linalg.eig(covariance[0:2,0:2])
    angle = np.arctan2(eigvecs[1,0], eigvecs[0,0]) * 360 / (2 * np.pi)
    width, height = 3 * np.sqrt(eigvals)

    ellipse = matplotlib.patches.Ellipse(xy=mean[0:2,3], width=width, height=height, angle=angle * 360 / (2 * np.pi), fill=False, color=color)
    ax.add_artist(ellipse)


def plot_trajectory_translation(trajectory, ax, palette='Blues'):
    xs, ys = trajectory[:, 0, 3], trajectory[:, 1, 3]

    sns.scatterplot(xs, ys, list(range(len(trajectory))), palette=palette, edgecolors=None)
    # ax.plot(trajectory[:,0,3], trajectory[:,1,3])
    ax.axis('equal')


def plot_trajectory_rotation(times, trajectory, ax):
    heading = np.empty(len(trajectory))
    for i, t in enumerate(trajectory):
        lie_vec = se3.log(t)
        heading[i] = lie_vec[5]

    ax.plot(times, -heading)

    ax.plot()


functions_of_plots = {
    'cov_on_density': plot_cov_against_density,
    'loss_on_time': plot_loss_on_time,
    'activation_matrix': plot_activation_matrix,
    'trajectory_evaluation': plot_trajectory_evaluation,
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
