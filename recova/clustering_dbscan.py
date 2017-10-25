#!/usr/bin/env python3
"""
Utilities to cluster a registration dataset using the DBSCAN algorithm.
"""

import argparse
import json
import numpy as np
from pyclustering.cluster import cluster_visualizer
import pyclustering.cluster.dbscan as dbscan
import sys

from pylie import se3_log
from recova.util import eprint
from recova.registration_dataset import lie_tensor_of_trails, lie_vectors_of_registrations


def dbscan_clustering(dataset, radius=0.005, n=12):
    """Augment dataset with clustering information using the dbscan algorithm"""

    if ('metadata' in dataset and
        'experiment' in dataset['metadata'] and
        dataset['metadata']['experiment'] == 'trail_batch'):
        lie_matrix = lie_tensor_of_trails(dataset)[-1]
    else:
        lie_matrix = lie_vectors_of_registrations(dataset)

    clustering = dbscan.dbscan(lie_matrix.tolist(), radius, n, True)
    clustering.process()

    statistics_dict = {
        'clustering': clustering.get_clusters(),
        'n_clusters': len(clustering.get_clusters()),
        'outliers': clustering.get_noise(),
        'outlier_ratio': len(clustering.get_noise()) / lie_matrix.shape[0]
    }

    if 'statistics' in dataset:
        dataset['statistics'].update(statistics_dict)
    else:
        dataset['statistics'] = statistics_dict

    original_metadata = (dataset['metadata'] if 'metadata' in dataset else {})
    metadata_dict = original_metadata.update({
        'clustering': {
            'algorithm': 'dbscan',
            'radius': radius,
            'n': n
        }
    })

    dataset['medata'] = metadata_dict

    return {
        'what': 'clustering',
        'metadata': metadata_dict,
        'statistics': {'n_clusters': len(clustering.get_clusters()),
                       'outliers': clustering.get_noise(),
                       'outlier_ratio': len(clustering.get_noise()) / lie_matrix.shape[0]},
        'data': clustering.get_clusters()
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
