#!/usr/bin/env python3

import argparse
import json
import numpy as np
import sys

from pylie import se3_log
from recova.covariance_of_registrations import distribution_of_registrations
from recova.registration_dataset import lie_vectors_of_registrations
from recova.util import eprint


def distance_of_cluster(dataset, cluster):
    filtered_dataset = filter_with_cluster(dataset, cluster)

    registrations = [x['result'] for x in dataset['data']]
    registrations = np.array(registrations)

    mean, covariance = distribution_of_registrations(registrations)

    perturbation = np.dot(np.linalg.inv(np.array(dataset['metadata']['ground_truth'])), mean)
    norm = np.linalg.norm(se3_log(perturbation))
    return norm

def find_central_cluster(dataset, clustering):
    """
    :arg dataset: A dataset as a facet.
    :arg clustering: A list of lists reprensenting the points indices
    :returns: The cluster itself (as a list of indices).
    """
    lie_vectors = lie_vectors_of_registrations(dataset)
    norms = np.linalg.norm(lie_vectors, axis=1)

    cluster_distances = list(map(lambda x: distance_of_cluster(dataset, x), clustering))
    eprint('Clustering distances: {}'.format(cluster_distances))

    if cluster_distances:
        best_cluster = clustering[np.argmin(cluster_distances)]
    else:
        best_cluster = []

    return best_cluster


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
