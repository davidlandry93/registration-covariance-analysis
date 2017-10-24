#!/usr/bin/env python3


import argparse
import json
import numpy as np
import sys

from recova.clustering_dbscan import lie_vectors_of_registrations
from pyevtk.hl import pointsToVTK
from pylie import se3_log
from recova.trails_to_vtk import lie_tensor_of_trails
from recova.util import POSITIVE_STRINGS, empty_to_none


def data_dict_of_registration_data(registration_data):
    data_dict = {}
    lie_vectors = positions_of_registration_data(registration_data)

    if 'statistics' in registration_data:
        if 'clustering' in registration_data['statistics']:
            cluster_of_points = np.empty(lie_vectors.shape[0])
            cluster_of_points[:] = np.NAN
            for i, cluster in enumerate(registration_data['statistics']['clustering']):
                for point_index in cluster:
                    cluster_of_points[point_index] = i + 1

            data_dict['cluster'] = np.ascontiguousarray(cluster_of_points)


        if 'outlier' in registration_data['statistics']:
            outlier_mask = np.zeros(lie_vectors.shape[0], dtype=np.int8)
            for index in registration_data['statistics']['outliers']:
                outlier_mask[index] = 1

            data_dict['outlier'] = np.ascontiguousarray(outlier_mask)

    print(data_dict)
    return data_dict


def positions_of_registration_data(reg_data, initial_estimates=False):
    if ('metadata' in reg_data and
        'experiment' in reg_data['metadata'] and
        'trail_batch' == reg_data['metadata']['experiment']):
        lie_vectors = lie_tensor_of_trails(reg_data)[(0 if initial_estimates else -1)]
    else:
        key = 'result'
        if initial_estimates:
            key = 'initial_estimate'

        lie_vectors = lie_vectors_of_registrations(registration_data, key=key)

    return lie_vectors


def results_to_vtk(registration_data, filename, dims, initial_estimates=False):
    lie_vectors = positions_of_registration_data(registration_data, initial_estimates)

    pointsToVTK(filename,
                np.ascontiguousarray(lie_vectors[:,dims[0]]),
                np.ascontiguousarray(lie_vectors[:,dims[1]]),
                np.ascontiguousarray(lie_vectors[:,dims[2]]),
                data = empty_to_none(data_dict_of_registration_data(registration_data)))

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Where to save the vtk file.')
    parser.add_argument('--dim1', type=int, default=0)
    parser.add_argument('--dim2', type=int, default=1)
    parser.add_argument('--dim3', type=int, default=2)
    parser.add_argument('--rotation', type=str, default='false', help='Print rotation dimensions instead of translation.')
    parser.add_argument('--initial-estimate', type=str, default='false', help='Print initial estimates instead of results.')
    args = parser.parse_args()

    json_data = json.load(sys.stdin)

    if args.rotation in POSITIVE_STRINGS:
        dims = (3,4,5)
    else:
        dims = (args.dim1, args.dim2, args.dim3)

    results_to_vtk(json_data, args.output, dims, initial_estimates=args.initial_estimate in POSITIVE_STRINGS)


if __name__ == '__main__':
    cli()
