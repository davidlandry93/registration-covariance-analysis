#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random

from lieroy import se3

import recov.datasets
import recov.registration_algorithm

import recova.clustering
from recova.covariance import CensiCovarianceComputationAlgorithm
from recova.descriptor.factory import descriptor_factory
from recova.learning.learning import model_from_file
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import parallel_map_progressbar, parallel_starmap_progressbar


def plot_trajectory_evaluation_mahalanobis(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='Path to registration database')
    parser.add_argument('location', type=str, help='Name of the location to work on.')
    parser.add_argument('dataset', type=str, help='Path to point cloud dataset.')
    parser.add_argument('learning_dataset', type=str)
    parser.add_argument('model', type=str, help='Path to covariance prediction model.')
    parser.add_argument('--n-ellipse', type=int, help='Plot covariance ellipses every n times.', default=10)
    parser.add_argument('--plot-begin', type=int, help='First index to plot', default=0)
    parser.add_argument('--plot-end', type=int, help='Last index to plot', default=-1)
    parser.add_argument('-j', '--n-cores', type=int, default=8)
    args = parser.parse_args(args)

    with open(args.learning_dataset) as f:
        learning_dataset = json.load(f)
        descriptor_algo = descriptor_factory(learning_dataset['metadata']['descriptor_config'])

    db = RegistrationPairDatabase(args.database)

    dataset = recov.datasets.KittiDataset(pathlib.Path(args.dataset))
    model = model_from_file(args.model, 'cello')

    gt_trajectory, sampled_trajectory, cum_covariances, censi_cum_cov = collect_trajectory_data(db, args.location, dataset, learning_dataset, model)

    fig, ax = plt.subplots()
    mahalanobis_plot(gt_trajectory, sampled_trajectory, cum_covariances, censi_cum_cov, ax)
    plt.show()


def collect_trajectory_data(db, location, pointcloud_dataset, descriptor_config, covariance_model, n_sampled_trajectories = 1):
    descriptor_algo = descriptor_factory(descriptor_config)

    pairs = db.registration_pairs()
    pairs = list(filter(lambda x: x.dataset == location and x.reference == x.reading - 1, pairs))

    gt_trajectory = build_trajectory_from_dataset(pointcloud_dataset)

    # Compute predicted covariance.
    predictions = predict_covariances(pairs, descriptor_algo, covariance_model)
    cum_covariances = make_cumulative_covariances(gt_trajectory, predictions)

    # Make sampled trajectory.
    # clustering_algo = recova.clustering.CenteredClusteringAlgorithm(radius=0.01, k=20, n_seed_init=32)
    # clustering_algo.rescale = True
    clustering_algo = recova.clustering.DensityThresholdClusteringAlgorithm(threshold=100, k=16)
    clustering_algo = recova.clustering.RegistrationPairClusteringAdapter(clustering_algo)

    if n_sampled_trajectories == 1:
        sampled_trajectory = make_sampled_trajectory(pairs, clustering_algo)
    else:
        sampled_trajectory = [make_sampled_trajectory(pairs, clustering_algo) for _ in range(n_sampled_trajectories)]

    # Compute the censi estimated covariances.
    icp_algo = recov.registration_algorithm.IcpAlgorithm()
    icp_algo.max_iteration_count = 60
    cov_algo = CensiCovarianceComputationAlgorithm(icp_algo, sensor_noise_std=0.01)
    censi_covariances = parallel_map_progressbar(cov_algo.compute, pairs)
    censi_cum_cov = make_cumulative_covariances(gt_trajectory, censi_covariances)

    return gt_trajectory, sampled_trajectory, cum_covariances, censi_cum_cov



def mahalanobis_plot(gt_trajectory, sampled_trajectory, predicted_cum_cov, censi_cum_cov, ax):
    pass


def build_trajectory_from_dataset(p_dataset, begin=0, p_end=-1):
    if p_end == -1:
        end = p_dataset.n_clouds()
    else:
        end = p_end

    print(begin)
    print(end)

    trajectory = np.empty((end - begin, 4, 4))
    trajectory[0] = np.identity(4)

    for i in range(1, end - begin):
        trajectory[i] = trajectory[i - 1] @ p_dataset.ground_truth(i + begin, i-1 + begin)

    return trajectory


def predict_covariance(pair, descriptor_algo):
    return descriptor_algo.compute(pair)


def predict_covariances(pairs, descriptor_algo, model):
    descriptors = parallel_starmap_progressbar(predict_covariance, [(pair, descriptor_algo) for pair in pairs])

    descriptors_np = np.empty((len(pairs), len(descriptor_algo.labels())))
    for i, descriptor in enumerate(descriptors):
        descriptors_np[i] = descriptor

    predictions = model.predict(descriptors_np)

    return predictions


def make_cumulative_covariances(trajectory, predictions):
    cum_covariances = np.empty((len(trajectory), 6, 6))
    cum_covariances[0] = np.zeros((6,6))
    for i in range(1, len(trajectory)):

        _, compound_covariance = se3.compound_poses(trajectory[i], predictions[i-1], trajectory[i-1], cum_covariances[i-1])

        cum_covariances[i] = compound_covariance

    return cum_covariances


def compute_clustering(pair, clustering_algo):
    return clustering_algo.compute(pair)


def make_sampled_trajectory(pairs, clustering_algo, n_cores=8):
    trajectory = np.empty((len(pairs) + 1, 4, 4))
    trajectory[0] = np.identity(4)

    clusterings = parallel_starmap_progressbar(compute_clustering, [(pair, clustering_algo) for pair in pairs], n_cores=n_cores)

    for i in range(len(pairs)):
        pair = pairs[i]
        results = pair.registration_results()

        t = random.choice(results[clusterings[i]])
        trajectory[i + 1] = trajectory[i] @ t

    return trajectory
