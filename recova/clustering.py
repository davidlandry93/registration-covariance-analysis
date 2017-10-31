#!/usr/bin/python3

import argparse
import json
import multiprocessing
import numpy as np
import sys

from recova.registration_dataset import points_to_vtk, positions_of_registration_data


def clusters_of_points(clustering, n_points):
    clusters = np.full(n_points, np.NAN)

    for i, cluster in enumerate(clustering):
        for point in cluster:
            clusters[point] = i

    return clusters


def to_vtk(dataset, clustering, output):
    """
    Make a vtk file showing a clustering on a certain dataset.
    :arg dataset: The full dataset to which the clustering is applied.
    :arg clustering: The clustering data row.
    """
    density = float(clustering['density'])
    clustering_data = clusters_of_points(clustering['clustering'], len(dataset))

    points_to_vtk(dataset[:,0:3], '{}_translation'.format(output, density), data={'clustering': np.ascontiguousarray(clustering_data)})
    points_to_vtk(dataset[:,3:6], '{}_rotation'.format(output, density), data={'clustering': np.ascontiguousarray(clustering_data)})


def batch_to_vtk(dataset, clustering_batch, output):
    """
    Make vtk files showing the evolution of clustering depending on a parameter.
    :arg dataset: The matrix of points on which the clustering was applied.
    :arg clustering_batch: The full clustering batch dataset.
    """

    with multiprocessing.Pool() as pool:
        pool.starmap(to_vtk, [(dataset, x, '{}_{}'.format(output, i)) for i, x in enumerate(clustering_batch['data'])])


def batch_to_vtk_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset on which the clustering is applied.', type=str)
    parser.add_argument('output', help='Prefix of the output vtk files.', type=str)
    args = parser.parse_args()

    clusterings = json.load(sys.stdin)
    with open(args.dataset) as dataset_file:
        dataset = json.load(dataset_file)

    points = positions_of_registration_data(dataset)

    batch_to_vtk(points, clusterings, args.output)

