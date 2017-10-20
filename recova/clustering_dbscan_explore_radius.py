#!/usr/bin/python3

import argparse
import copy
import itertools
import json
import multiprocessing.pool
import numpy as np
import sys
import time
import threading

from clustering_dbscan import dbscan_clustering
from clustering_to_vtk import clustering_to_vtk


def save_clustered_data(clustered_data):
    filename = '{}{}{}_dbscan{:.8f}'.format(
        clustered_data['metadata']['dataset'],
        clustered_data['metadata']['reading'],
        clustered_data['metadata']['reference'],
        clustered_data['metadata']['clustering']['dbscan_radius']).replace('.', '')

    clustering_to_vtk(clustered_data, filename + '_translation', (0,1,2))
    clustering_to_vtk(clustered_data, filename + '_rotation', (3,4,5))


def run_one_clustering_thread(registration_data, radius):
    print('Clustering with radius {}'.format(radius))

    copied_data = copy.copy(registration_data)
    dbscan_clustering(copied_data, radius)

    save_clustered_data(copied_data)
    print('Done clustering with radius {}'.format(radius))


if __name__ == '__main__':
    json_dataset = json.load(sys.stdin)

    parser = argparse.ArgumentParser()
    parser.add_argument('begin', type=float)
    parser.add_argument('end', type=float)
    parser.add_argument('delta', type=float)
    args = parser.parse_args()

    pool = multiprocessing.pool.Pool()
    pool.starmap(run_one_clustering_thread,
                 zip(itertools.repeat(json_dataset), np.arange(args.begin, args.end, args.delta)),
                 chunksize=1)
    pool.close()
