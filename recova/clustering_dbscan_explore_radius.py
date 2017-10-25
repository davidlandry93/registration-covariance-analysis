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

from recova.clustering_dbscan import dbscan_clustering
from recova.registration_dataset import dataset_to_vtk
from recova.util import eprint


def save_clustered_data(clustered_data):
    filename = '{}{}{}_dbscan{:.8f}'.format(
        clustered_data['metadata']['dataset'],
        clustered_data['metadata']['reading'],
        clustered_data['metadata']['reference'],
        clustered_data['metadata']['clustering']['radius']).replace('.', '')

    dataset_to_vtk(clustered_data, filename + '_translation')
    dataset_to_vtk(clustered_data, filename + '_rotation', (3,4,5))


def run_one_clustering_thread(i, registration_data, radius, n=12):
    eprint('Clustering with radius {}'.format(radius))

    copied_data = copy.copy(registration_data)
    clustering = dbscan_clustering(copied_data, radius)
    save_clustered_data(copied_data)

    eprint('Done clustering with radius {}'.format(radius))

    return clustering


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('begin', type=float)
    parser.add_argument('end', type=float)
    parser.add_argument('delta', type=float)
    args = parser.parse_args()

    json_dataset = json.load(sys.stdin)

    radiuses = np.arange(args.begin, args.end, args.delta)

    with multiprocessing.Pool() as pool:
        clusterings = pool.starmap(run_one_clustering_thread,
                                   [(x, json_dataset, radiuses[x]) for x in range(len(radiuses))],
                                   chunksize=1)
        json.dump(clusterings, sys.stdout)

if __name__ == '__main__':
    cli()
