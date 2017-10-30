#!/usr/bin/env python3
"""
Utilities to cluster a registration dataset using the DBSCAN algorithm.
"""

import argparse
import json
import numpy as np
import sys
import time

import pyclustering.cluster.dbscan as dbscan

from pylie import se3_log
from recova.util import eprint
from recova.registration_dataset import lie_tensor_of_trails, lie_vectors_of_registrations



def dbscan_clustering(dataset, radius=0.005, n=12):
    """
    Return a data row describing a clustering that was run on dataset with the given parameters.
    """

    if ('metadata' in dataset and
        'experiment' in dataset['metadata'] and
        dataset['metadata']['experiment'] == 'trail_batch'):
        lie_matrix = lie_tensor_of_trails(dataset)[-1]
    else:
        lie_matrix = lie_vectors_of_registrations(dataset)

    lie_matrix[:,0:3] = rescale_hypersphere(lie_matrix[:,0:3], 2*np.pi)

    clustering = dbscan.dbscan(lie_matrix.tolist(), radius, n, True)

    start = time.clock()
    clustering.process()
    computation_time = time.clock() - start

    return {
        'clustering': clustering.get_clusters(),
        'n_clusters': len(clustering.get_clusters()),
        'outliers': clustering.get_noise(),
        'outlier_ratio': len(clustering.get_noise()) / len(lie_matrix),
        'computation_time': computation_time,
        'density': radius * len(lie_matrix)
    }


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=float, default=0.005,
                        help='Radius that must contain enough neighbors for a point to be a core point.')
    parser.add_argument('--n', type=float, default=12,
                        help='Number of neighbours that must be contained within the radius for a point to be a core point.')
    args = parser.parse_args()

    json_dataset = json.load(sys.stdin)

    dbscan_clustering(json_dataset, args.radius, args.n)

    json.dump(json_dataset, sys.stdout)

    cluster_sizes = sorted(list(map(len, json_dataset['statistics']['clustering'])), reverse=True)

    eprint(cluster_sizes)
    eprint('N clusters: {}'.format(json_dataset['statistics']['n_clusters']))
    eprint('Outlier ratio: {}'.format(json_dataset['statistics']['outlier_ratio']))
    eprint('Cluster sizes: {}'.format(cluster_sizes))


if __name__ == '__main__':
    cli()
