#!/usr/bin/env python3

import argparse
import json
import numpy as np
import sys

from lieroy.parallel import FunctionWrapper
from pylie import se3_log
from recova.covariance_of_registrations import distribution_of_registrations
from recova.registration_dataset import lie_vectors_of_registrations
from recova.util import eprint, dataset_to_registrations

def index_of_closest_to_ground_truth(dataset):
    """
    Find the index of the point in the dataset that is the closest to ground truth.
    :arg dataset: The registration dataset as a facet.
    """
    gt = np.array(dataset['metadata']['ground_truth'])
    inv_of_gt = np.linalg.inv(gt)

    id_of_min = None
    min_distance = np.inf
    for i, registration in enumerate(dataset['data']):
        print(registration)
        reg = np.array(registration['result'])
        distance_to_gt = np.linalg.norm(se3_log(np.dot(inv_of_gt, reg)))

        if distance_to_gt < min_distance:
            id_of_min = i
            min_distance = distance_to_gt

    eprint('Min distance to ground truth: {}'.format(min_distance))

    return id_of_min

def distance_of_cluster(dataset, cluster):
    filtered_dataset = filter_with_cluster(dataset, cluster)

    registrations = dataset_to_registrations(dataset)

    inv_of_gt = np.linalg.inv(np.array(dataset['metadata']['ground_truth']))

    min_distance = np.inf
    for point in cluster:
        reg = registrations[point]
        distance_to_gt = np.linalg.norm(se3_log(np.dot(inv_of_gt, reg)))
        if distance_to_gt < min_distance:
            min_distance = distance_to_gt

    eprint('Cluster has {} distance to GT.'.format(min_distance))

    return min_distance

def find_central_cluster(dataset, clustering):
    """
    :arg dataset: A dataset as a facet.
    :arg clustering: A list of lists reprensenting the points indices
    :returns: The cluster itself (as a list of indices).
    """

    if len(clustering) == 1:
        return clustering[0], distance_of_cluster(dataset, clustering[0])

    closest_point = index_of_closest_to_ground_truth(dataset)

    lie_vectors = lie_vectors_of_registrations(dataset)
    norms = np.linalg.norm(lie_vectors, axis=1)


    cluster_distances = list(map(lambda x: distance_of_cluster(dataset, x), clustering))
    eprint('Clustering distances: {}'.format(cluster_distances))

    if cluster_distances:
        best_cluster = clustering[np.argmin(cluster_distances)]
    else:
        best_cluster = []

    return best_cluster, np.min(cluster_distances)


def filter_with_cluster(dataset, cluster):
    new_data = []
    for i in cluster:
        new_data.append(dataset['data'][i])

    new_dataset = dataset.copy()
    new_dataset['data'] = new_data

    return new_dataset


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clustering', type=str, help='Name of clustering file to use', default='')
    args = parser.parse_args()

    dataset = json.load(sys.stdin)

    if args.clustering:
        with open(args.clustering) as clustering_file:
            clustering = json.load(clustering_file)['data']
    else:
        clustering = dataset['statistics']['clustering']

    central_cluster_ids = find_central_cluster(dataset, clustering)

    size_before = len(dataset['data'])
    central_cluster_points = filter_with_cluster(dataset, central_cluster_ids)
    eprint('{}% of the data was part of the central cluster.'.format((len(dataset['data']) / size_before)*100.))

    json.dump(dataset, sys.stdout)


if __name__ == '__main__':
    cli()
