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

from recova.clustering import clustering_algorithm_factory, compute_distribution
from recova.registration_dataset import lie_vectors_of_registrations
from recova.util import eprint


def rescale_hypersphere(points, radius):
    norms = np.linalg.norm(points, axis=1)
    max_distance = np.max(norms)

    points = points * radius / max_distance

    return points


def run_one_clustering_thread(algo, i, registration_data, density, n=12, scaling_of_translation=False):
    eprint('Clustering with density {}'.format(density))

    radius = density / len(registration_data['data'])

    if scaling_of_translation:
        lie_vectors[:,0:3] = rescale_hypersphere(lie_vectors[:,0:3], scaling_of_translation)

    clustering = algo(registration_data, radius)

    clustering_with_distribution = compute_distribution(registration_data, clustering)

    eprint('Done clustering with radius {} (density {})'.format(radius, density))

    return clustering_with_distribution


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', type=str, help='Clustering algorithm to use')
    parser.add_argument('begin', type=float)
    parser.add_argument('end', type=float)
    parser.add_argument('n_samples', type=float)
    parser.add_argument('--scale_translations', action='store_true')
    args = parser.parse_args()

    json_dataset = json.load(sys.stdin)

    densities = np.linspace(args.begin, args.end, args.n_samples)

    if args.scale_translations:
        translation_scaling = 2*3.1416
    else:
        translation_scaling = False

    clustering_algorithm = clustering_algorithm_factory(args.algo)

    with multiprocessing.Pool() as pool:
        clusterings = pool.starmap(run_one_clustering_thread,
                                   [(clustering_algorithm, x, json_dataset, densities[x], translation_scaling) for x in range(len(densities))],
                                   chunksize=1)

    facet = {
        'what': 'clusterings',
        'metadata': {
            'translations_scaling': translation_scaling,
        },
        'statistics': {},
        'data': clusterings,
        'facets': []
    }

    json.dump(facet, sys.stdout)

if __name__ == '__main__':
    cli()
