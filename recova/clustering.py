#!/usr/bin/python3

import argparse
import io
import json
import multiprocessing
import numpy as np
import subprocess
import sys
import time

import pyclustering.cluster.dbscan as dbscan

from recova.covariance_of_registrations import distribution_of_registrations
from recova.registration_dataset import points_to_vtk, positions_of_registration_data, registrations_of_dataset, lie_vectors_of_registrations
from recova.find_center_cluster import find_central_cluster, filter_with_cluster
from recova.util import eprint

import recova_core



def inverse_of_cluster(cluster, size_of_dataset):
    """Returns a list of points not in cluster."""
    sorted_cluster = sorted(cluster)
    inverse = []
    for i in range(size_of_dataset-1, -1, -1):
        if sorted_cluster and sorted_cluster[-1] != i:
            inverse.append(i)
        elif sorted_cluster:
            sorted_cluster.pop()
        else:
            inverse.append(i)

    return inverse


def centered_clustering(dataset, radius, n=12, seed=np.zeros(6)):
    """
    :arg dataset: A nd array containing the points to cluster.
    :returns: The result of the centered clustering algorithm, as a facet.
    """

    # TODO Replace the call to raw_centered_clustering with a call to the cpp library recova_core
    center_cluster = raw_centered_clustering(dataset, radius, n, seed)

    clustering_row = {
        'clustering': [center_cluster],
        'n_clusters': 1,
        'radius': radius,
        'n': n,
        'outliers': inverse_of_cluster(center_cluster, len(dataset)),
        'outlier_ratio': 1.0 - (len(center_cluster) / len(dataset)),
    }

    eprint('{} radius'.format(radius))
    eprint('{} outliers'.format(len(clustering_row['outliers'])))
    eprint('{} inliers'.format(len(center_cluster)))


    return clustering_row


def raw_centered_clustering(dataset, radius, n=12, seed=np.zeros(6)):
    """
    :arg dataset: The dataset to cluster (as a numpy matrix).
    :arg radius: The radius in which we have to have enough neighbours to be a core point.
    :arg n: The number of points that have to be within the radius to be a core point.
    :returns: The indices of the points that are inside the central cluster as a list.
    """
    strings_of_seed = list(map(str, seed.tolist()))
    command = 'centered_clustering -radius {} -n {} -seed {}'.format(radius, n, ','.join(strings_of_seed))
    eprint(command)
    stream = io.StringIO()
    json.dump(dataset.tolist(), stream)
    response = subprocess.run(command,
                              input=json.dumps(dataset.tolist()),
                              stdout=subprocess.PIPE,
                              shell=True,
                              universal_newlines=True)

    return json.loads(response.stdout)


def dbscan_clustering(dataset, radius=0.005, n=12, seed=None):
    """
    :arg dataset: A np array describing the points to cluster.
    :returns: A datarow representing the clustering.
    """
    clustering = dbscan.dbscan(dataset.tolist(), radius, n, True)

    start = time.clock()
    clustering.process()
    computation_time = time.clock() - start

    return {
        'clustering': clustering.get_clusters(),
        'n_clusters': len(clustering.get_clusters()),
        'outliers': clustering.get_noise(),
        'outlier_ratio': len(clustering.get_noise()) / len(dataset),
        'computation_time': computation_time,
        'radius': radius,
        'n': n
    }


def clustering_algorithm_factory(algo_name):
    algo_dict = {
        'centered': centered_clustering,
        'dbscan':  dbscan_clustering
    }
    return algo_dict[algo_name]


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


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', type=str, default='centered', help='The name of the clustering algorithm to use. {centered,dbscan}')
    parser.add_argument('--radius', type=float, default=0.005,
                        help='Radius that must contain enough neighbors for a point to be a core point.')
    parser.add_argument('--n', type=float, default=12,
                        help='Number of neighbours that must be contained within the radius for a point to be a core point.')
    args = parser.parse_args()

    json_dataset = json.load(sys.stdin)
    algo = clustering_algorithm_factory(args.algo)

    clustering = algo(json_dataset, args.radius, args.n)

    clustering_with_distribution = compute_distribution(json_dataset, clustering)

    json.dump(clustering_with_distribution, sys.stdout)

    cluster_sizes = sorted(list(map(len, clustering['clustering'])), reverse=True)
    eprint(cluster_sizes)
