#!/usr/bin/python3

import argparse
import copy
import itertools
import json
import multiprocessing
import numpy as np
import sys
import time
import threading

from lieroy import se3, parallel

from recova.clustering import CenteredClusteringAlgorithm, compute_distribution
from recova.registration_dataset import lie_vectors_of_registrations, positions_of_registration_data
from recova.util import eprint, rescale_hypersphere





def run_one_clustering_thread(radius, k, registration_data, rescale_data=False, n_seed_init=100, seed_selector='localized'):
    eprint('Clustering with radius {}'.format(radius))

    var_translation = float(registration_data['metadata']['var_translation'])

    lie_vectors = positions_of_registration_data(registration_data)

    ground_truth = np.array(registration_data['metadata']['ground_truth'])

    algo = CenteredClusteringAlgorithm(radius, k, n_seed_init)
    algo.rescale = rescale_data
    algo.n_seed_init = n_seed_init
    algo.seed_selector = seed_selector

    se3exp = parallel.FunctionWrapper('log', 'lieroy.se3')
    clustering = algo.cluster(lie_vectors, seed=se3exp(ground_truth))

    clustering_with_distribution = compute_distribution(registration_data, clustering)

    eprint('Done clustering with radius {} '.format(radius))

    return clustering_with_distribution


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('begin', type=float)
    parser.add_argument('end', type=float)
    parser.add_argument('n_samples', type=float)
    parser.add_argument('--scale_translations', action='store_true')
    parser.add_argument('--n_neighbors', type=int, default=12)
    parser.add_argument('--n_seed_init', type=int, default=100, help='Number of seed points to consider during the initialization.')
    parser.add_argument('--seed_selector', type=str, default='localized')
    args = parser.parse_args()

    json_dataset = json.load(sys.stdin)

    radii = np.linspace(args.begin, args.end, args.n_samples)


    with multiprocessing.Pool(2) as pool:
        clusterings = pool.starmap(run_one_clustering_thread,
                                   [(x, args.n_neighbors, json_dataset, args.scale_translations, args.n_seed_init, args.seed_selector) for x in radii],
                                   chunksize=1)

    metadata_dict = json_dataset['metadata']
    metadata_dict['translations_scaling'] = args.scale_translations


    facet = {
        'what': 'clusterings',
        'metadata': metadata_dict,
        'statistics': {},
        'data': clusterings,
        'facets': []
    }

    json.dump(facet, sys.stdout)

if __name__ == '__main__':
    cli()
