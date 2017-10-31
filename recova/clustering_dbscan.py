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
    :arg dataset: A numpy matrix containing the N-D points to cluster.
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
        'density': radius * len(dataset)
    }


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=float, default=0.005,
                        help='Radius that must contain enough neighbors for a point to be a core point.')
    parser.add_argument('--n', type=float, default=12,
                        help='Number of neighbours that must be contained within the radius for a point to be a core point.')
    args = parser.parse_args()

    json_dataset = json.load(sys.stdin)
    lie_vectors = lie_vectors_of_registrations(json_dataset)
    clustering = dbscan_clustering(lie_vectors, args.radius, args.n)

    json.dump(clustering, sys.stdout)

    cluster_sizes = sorted(list(map(len, clustering['clustering'])), reverse=True)

    eprint(cluster_sizes)
    eprint('N clusters: {}'.format(clustering['n_clusters']))
    eprint('Outlier ratio: {}'.format(clustering['outlier_ratio']))
    eprint('Cluster sizes: {}'.format(cluster_sizes))


if __name__ == '__main__':
    cli()
