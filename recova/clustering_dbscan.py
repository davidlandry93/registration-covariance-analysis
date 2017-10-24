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
from recova.trails_to_vtk import lie_tensor_of_trails

def lie_vectors_of_registrations(json_data, key='result'):
    """
    Outputs the lie vectors of a json registration dataset.

    :param json_data: A full registration dataset.
    :param key: The key of the matrix to evaluate, inside the registration result.
    :returns: A Nx6 numpy matrix containing the lie algebra representation of the results.
    """
    lie_results = np.empty((len(json_data['data']), 6))
    for i, registration in enumerate(json_data['data']):
        m = np.array(registration[key])

        try:
            lie_results[i,:] = se3_log(m)
        except RuntimeError:
            lie_results[i,:] = np.zeros(6)
            eprint('Warning: failed conversion to lie algebra of matrix {}'.format(m))

    return lie_results


def dbscan_clustering(dataset, radius=0.005, n=12):
    """Augment dataset with clustering information using the dbscan algorithm"""

    if ('metadata' in dataset and
        'experiment' in dataset['metadata'] and
        dataset['metadata']['experiment'] == 'trail_batch'):
        lie_matrix = lie_tensor_of_trails(dataset)[-1]
    else:
        lie_matrix = lie_vectors_of_registrations(registration_dataset)

    print(lie_matrix)

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

    metadata_dict = {
        'clustering': {
            'algorithm': 'dbscan',
            'dbscan_radius': radius,
            'dbscan_n': n
        }
    }

    if 'metadata' in dataset:
        dataset['metadata'].update(metadata_dict)
    else:
        dataset['metadata'] = metadata_dict

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
