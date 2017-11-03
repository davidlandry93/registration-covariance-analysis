#!/usr/bin/python3

import argparse
import json
import multiprocessing
import numpy as np
import sys

from recova.covariance_of_registrations import distribution_of_registrations
from recova.registration_dataset import points_to_vtk, positions_of_registration_data, registrations_of_dataset
from recova.find_center_cluster import find_central_cluster, filter_with_cluster


def clusters_of_points(clustering, n_points):
    clusters = np.full(n_points, np.NAN)

    for i, cluster in enumerate(clustering):
        for point in cluster:
            clusters[point] = i

    return clusters


def to_vtk(dataset, clustering, output):
    """
    Make a vtk file showing a clustering on a certain dataset.
    :arg dataset: The full dataset to which the clustering is applied.
    :arg clustering: The clustering data row.
    """
    density = float(clustering['density'])
    clustering_data = clusters_of_points(clustering['clustering'], len(dataset))

    points_to_vtk(dataset[:,0:3], '{}_translation'.format(output, density), data={'clustering': np.ascontiguousarray(clustering_data)})
    points_to_vtk(dataset[:,3:6], '{}_rotation'.format(output, density), data={'clustering': np.ascontiguousarray(clustering_data)})


def batch_to_vtk(dataset, clustering_batch, output):
    """
    Make vtk files showing the evolution of clustering depending on a parameter.
    :arg dataset: The matrix of points on which the clustering was applied.
    :arg clustering_batch: The full clustering batch dataset.
    """

    with multiprocessing.Pool() as pool:
        pool.starmap(to_vtk, [(dataset, x, '{}_{}'.format(output, i)) for i, x in enumerate(clustering_batch['data'])])


def distribution_of_cluster(dataset, cluster):
    new_dataset = dataset.copy()
    new_dataset = filter_with_cluster(new_dataset, cluster)

    registrations = registrations_of_dataset(new_dataset)

    if len(registrations) != 0:
        mean, covariance = distribution_of_registrations(registrations)
    else:
        mean = np.identity(4)
        covariance = np.zeros((6,6))

    return mean, covariance


def compute_distribution(dataset, clustering):
    """
    :arg cluster: The data row on which we want to compute the distribution of the central cluster.
    :returns: A new data row, this time with a distribution attached.
    """
    central_cluster = find_central_cluster(dataset, clustering['clustering'])
    mean, covariance = distribution_of_cluster(dataset, central_cluster)

    new_clustering = clustering.copy()
    new_clustering['mean_of_central'] = mean.tolist()
    new_clustering['covariance_of_central'] = covariance.tolist()
    new_clustering['central_cluster_size'] = len(central_cluster)
    new_clustering['central_cluster'] = central_cluster

    return new_clustering


def compute_distributions(dataset, clustering_batch):
    """
    Compute the distribution of the central cluster for clusterings.
    :arg dataset: The registration dataset the clustering was computed on.
    :arg clustering_batch: The collection of clustering (as a facet).
    :returns: The same facet, but the clustering data rows now have a distribution attached to them.
    """
    clusterings = clustering_batch.copy()

    new_clusterings = []
    for clustering in clustering_batch['data']:
        new_clusterings.append(compute_distribution(dataset, clustering))

    clusterings['data'] = new_clusterings
    return clusterings


def compute_distributions_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset on which the clustering is applied.', type=str)
    args = parser.parse_args()

    clustering_batch = json.load(sys.stdin)
    with open(args.dataset) as dataset_file:
        dataset = json.load(dataset_file)

    batch_with_distributions = compute_distributions(dataset, clustering_batch)
    json.dump(batch_with_distributions, sys.stdout)


def batch_to_vtk_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset on which the clustering is applied.', type=str)
    parser.add_argument('output', help='Prefix of the output vtk files.', type=str)
    args = parser.parse_args()

    clusterings = json.load(sys.stdin)
    with open(args.dataset) as dataset_file:
        dataset = json.load(dataset_file)

    points = positions_of_registration_data(dataset)

    batch_to_vtk(points, clusterings, args.output)

