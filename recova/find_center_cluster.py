#!/usr/bin/env python3

import json
import numpy as np
import sys

from clustering_dbscan import lie_vectors_of_registrations
from util import eprint

def find_central_cluster(dataset):
    lie_vectors = lie_vectors_of_registrations(dataset)
    norms = np.linalg.norm(lie_vectors, axis=1)

    cluster_distances = list(map(lambda x: norms[x[0]], dataset['statistics']['clustering']))
    best_cluster = np.argmin(cluster_distances)

    return best_cluster

def filter_one_cluster(dataset, cluster):
    new_data = []
    for i in dataset['statistics']['clustering'][cluster]:
        new_data.append(dataset['data'][i])

    dataset['data'] = new_data


if __name__ == '__main__':
    dataset = json.load(sys.stdin)


    central_cluster = find_central_cluster(dataset)

    size_before = len(dataset['data'])
    filter_one_cluster(dataset, central_cluster)
    eprint('{}% of the data was part of the central cluster.'.format((len(dataset['data']) / size_before)*100.))

    dataset['']

    json.dump(dataset, sys.stdout)
