#!/usr/bin/env python3

import argparse
import json
import numpy as np
import sys

from recova.registration_dataset import lie_vectors_of_registrations
from recova.util import eprint

def find_central_cluster(dataset, clustering):
    lie_vectors = lie_vectors_of_registrations(dataset)
    norms = np.linalg.norm(lie_vectors, axis=1)

    cluster_distances = list(map(lambda x: norms[x[0]], clustering))

    if cluster_distances:
        best_cluster = clustering[np.argmin(cluster_distances)]
    else:
        best_cluster = []

    return best_cluster

def filter_with_cluster(dataset, cluster):
    new_data = []
    for i in cluster:
        new_data.append(dataset['data'][i])

    dataset['data'] = new_data

    return dataset

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
