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
from recova.distribution_to_vtk_ellipsoid import distribution_to_vtk_ellipsoid
from recova.registration_dataset import points_to_vtk, positions_of_registration_data, registrations_of_dataset, lie_vectors_of_registrations, data_dict_of_registration_data
from recova.find_center_cluster import find_central_cluster
from recova.util import eprint, rescale_hypersphere, englobing_radius

from lieroy.parallel import FunctionWrapper

import recova_core

se3log = FunctionWrapper('log', 'lieroy.se3')
se3exp = FunctionWrapper('exp', 'lieroy.se3')

class ClusteringAlgorithm:
    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError('Clustering Algorithms must implement __repr__')

    def cluster(dataset):
        raise NotImplementedError('Clustering Algorithms must implement the cluster method')


class CenteredClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, radius=0.2, k=12, n_seed_init=100):
        self.radius = radius
        self.k = k
        self.n_seed_init = n_seed_init
        self.rescale = False
        self.seed_selector = 'greedy'
        self.logging = False

    def __repr__(self):
        return 'centered_{:.5f}_{}_{}'.format(self.radius, self.k, self.rescale)

    def cluster(self, dataset, seed=np.array([0., 0., 0., 0., 0., 0.])):
        if self.rescale:
            # Rescale translation and rotation separately so that they don't crush one another.
            radius_translation = englobing_radius(dataset[:, 0:3], 90.0)
            radius_rotation = englobing_radius(dataset[:, 3:6], 90.0)

            dataset[:,0:3] = dataset[:,0:3] / radius_translation
            dataset[:,3:6] = dataset[:,3:6] / radius_rotation

            seed[0:3] = seed[0:3] / radius_translation
            seed[3:6] = seed[3:6] / radius_rotation


        center_cluster = raw_centered_clustering(dataset, self.radius, self.k, seed, self.n_seed_init, seed_selector=self.seed_selector, logging=self.logging)

        clustering_row = {
            'clustering': [center_cluster],
            'n_clusters': 1,
            'radius': self.radius,
            'n': self.k,
            'outliers': inverse_of_cluster(center_cluster, len(dataset)),
            'outlier_ratio': 1.0 - (len(center_cluster) / len(dataset)),
        }

        eprint('{} radius'.format(self.radius))
        eprint('{} outliers'.format(len(clustering_row['outliers'])))
        eprint('{} inliers'.format(len(center_cluster)))


        return clustering_row

class DBSCANClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, radius=0.2, k=12):
        self.radius = radius
        self.k = k

    def __repr__(self):
        return 'dbscan_{:.5f}_{}'.format(self.radius, self.k)

    def cluster(self, dataset):
        clustering = dbscan.dbscan(dataset.tolist(), self.radius, self.k, True)

        start = time.clock()
        clustering.process()
        computation_time = time.clock() - start

        return {
            'clustering': clustering.get_clusters(),
            'n_clusters': len(clustering.get_clusters()),
            'outliers': clustering.get_noise(),
            'outlier_ratio': len(clustering.get_noise()) / len(dataset),
            'computation_time': computation_time,
            'radius': self.radius,
            'n': self.k
        }


class IdentityClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self):
        pass

    def __repr__(self):
        return 'identity_clustering'

    def cluster(self, dataset, seed=None):
        return {
            'clustering': [list(range(0, len(dataset)))],
            'n_clusters': 1,
            'outliers': [],
            'outlier_ratio': 0.0,
        }




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


def raw_centered_clustering(dataset, radius, n=12, seed=np.zeros(6), n_seed_init=100, seed_selector='greedy', logging=False):
    """
    :arg dataset: The dataset to cluster (as a numpy matrix).
    :arg radius: The radius in which we have to have enough neighbours to be a core point.
    :arg n: The number of points that have to be within the radius to be a core point.
    :returns: The indices of the points that are inside the central cluster as a list.
    """
    strings_of_seed = list(map(str, seed.tolist()))
    string_of_seed = ','.join(strings_of_seed)

    command = 'centered_clustering -seed_selector {} -k {} -radius {} -seed {} {}'.format(seed_selector, n, radius, string_of_seed, ('--pointcloud_log' if logging else '--nopointcloud_log'))

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


def clustering_algorithm_factory(algo_name):
    algo_dict = {
        'centered': CenteredClusteringAlgorithm,
        'dbscan':  DBSCANClusteringAlgorithm,
        'identity': IdentityClusteringAlgorithm
    }
    return algo_dict[algo_name]()


def clusters_of_points(clustering, n_points):
    clusters = np.full(n_points, np.NAN)

    for i, cluster in enumerate(clustering):
        for point in cluster:
            clusters[point] = i

    return clusters


def to_vtk(dataset, clustering, output, center_around_gt=False):
    """
    Make a vtk file showing a clustering on a certain dataset.
    :arg dataset: The full dataset to which the clustering is applied.
    :arg clustering: The clustering data row.
    """
    points = positions_of_registration_data(dataset)

    if center_around_gt:
        points = center_lie_around_t(points, np.array(dataset['metadata']['ground_truth']))

    radius = float(clustering['radius'])
    clustering_data = clusters_of_points(clustering['clustering'], len(points))

    data_dict = data_dict_of_registration_data(dataset)
    data_dict['clustering'] = np.ascontiguousarray(clustering_data)

    points_to_vtk(points[:,0:3], '{}_translation'.format(output, radius), data=data_dict)
    points_to_vtk(points[:,3:6], '{}_rotation'.format(output, radius), data=data_dict)

    mean = np.array(clustering['mean_of_central'])
    mean = se3log(mean)
    covariance = np.array(clustering['covariance_of_central'])

    distribution_to_vtk_ellipsoid(mean[0:3], covariance[0:3, 0:3], '{}_translation_ellipsoid'.format(output))
    distribution_to_vtk_ellipsoid(mean[3:6], covariance[3:6, 3:6], '{}_rotation_ellipsoid'.format(output))


def center_lie_around_t(points, T):
    """
    Recenter a collection of lie algebra vectors around a new 4x4 T.
    """
    inv_of_T = np.linalg.inv(T)

    for i in range(len(points)):
        points[i] = se3log(np.dot(se3exp(points[i]), inv_of_T))

    return points


def batch_to_vtk(dataset, clustering_batch, output, center_around_gt=False):
    """
    Make vtk files showing the evolution of clustering depending on a parameter.
    :arg dataset: The matrix of points on which the clustering was applied.
    :arg clustering_batch: The full clustering batch dataset.
    """

    with multiprocessing.Pool(3) as pool:
        pool.starmap(to_vtk, [(dataset, x, '{}_{}'.format(output, i), center_around_gt) for i, x in enumerate(clustering_batch['data'])])


def distribution_of_cluster(dataset, cluster):
    registrations = registrations_of_dataset(dataset)
    registrations = registrations[cluster]


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

    central_cluster, cluster_distance = find_central_cluster(dataset, clustering['clustering'])
    mean, covariance = distribution_of_cluster(dataset, central_cluster)

    new_clustering = clustering.copy()
    new_clustering['mean_of_central'] = mean.tolist()
    new_clustering['covariance_of_central'] = covariance.tolist()
    new_clustering['central_cluster_size'] = len(central_cluster)
    new_clustering['central_cluster'] = central_cluster
    new_clustering['cluster_distance'] = cluster_distance

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


def clustering_to_vtk_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset on which the clustering is applied', type=str)
    parser.add_argument('output', help='Prefix of the output vtk files', type=str)
    parser.add_argument('--center_around_gt', action='store_true')
    args = parser.parse_args()

    print('Loading from stdin')
    clustering = json.load(sys.stdin)

    print('Loading dataset')
    with open(args.dataset) as dataset_file:
        dataset = json.load(dataset_file)

    to_vtk(dataset, clustering, args.output, center_around_gt=args.center_around_gt)




def batch_to_vtk_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset on which the clustering is applied.', type=str)
    parser.add_argument('output', help='Prefix of the output vtk files.', type=str)
    parser.add_argument('--center_around_gt', help='Center the results around the ground truth', action='store_true')
    args = parser.parse_args()

    print('Loading from stdin')
    clusterings = json.load(sys.stdin)

    print('Loading from file')
    with open(args.dataset) as dataset_file:
        dataset = json.load(dataset_file)

    batch_to_vtk(dataset, clusterings, args.output, args.center_around_gt)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', type=str, default='centered', help='The name of the clustering algorithm to use. {centered,dbscan}')
    parser.add_argument('--radius', type=float, default=0.005,
                        help='Radius that must contain enough neighbors for a point to be a core point.')
    parser.add_argument('--n', type=int, default=12,
                        help='Number of neighbours that must be contained within the radius for a point to be a core point.')
    parser.add_argument('--rescale_data', action='store_true', help='Scale rotations and translations so that the rotations live within a 1.0 radius sphere and the translations live within a 1.0 radius sphere.')
    parser.add_argument('--seed_selector', type=str, help='The seed selection strategy to use. <greedy|centered>', default='greedy')
    args = parser.parse_args()

    json_dataset = json.load(sys.stdin)
    algo = clustering_algorithm_factory(args.algo)

    data = lie_vectors_of_registrations(json_dataset)

    ground_truth = np.array(json_dataset['metadata']['ground_truth'])

    algo.radius = args.radius
    algo.k = args.n
    algo.rescale = args.rescale_data
    algo.seed_selector = args.seed_selector
    algo.logging = True
    clustering = algo.cluster(data, seed=se3log(ground_truth))

    clustering_with_distribution = compute_distribution(json_dataset, clustering)

    json.dump(clustering_with_distribution, sys.stdout)

    cluster_sizes = sorted(list(map(len, clustering['clustering'])), reverse=True)
    eprint(cluster_sizes)
