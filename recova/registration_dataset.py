#!/usr/bin/env python3

import json
import numpy as np
import sys

from pyevtk.hl import pointsToVTK
from pylie import se3_log

from recova.density import density_of_points
from recova.util import empty_to_none, eprint


def dataset_to_vtk(dataset, filename, dims=(0,1,2)):
    positions = positions_of_registration_data(dataset)
    data = empty_to_none(data_dict_of_registration_data(dataset))

    points_to_vtk(positions[:,dims], filename, data=data)


def registrations_of_dataset(dataset, key='result'):
    """
    A numpy array containing all the registration results contained in a dataset.
    The registration results are in matrix form.
    """
    registrations = np.empty((len(dataset['data']), 4, 4))
    for i, registration in enumerate(dataset['data']):
        registrations[i] = registration['result']

    return registrations




def lie_vectors_of_registrations(json_data, key='result', prealignment=np.identity(4)):
    """
    Outputs the lie vectors of a json registration dataset.

    :param json_data: A full registration dataset.
    :param key: The key of the matrix to evaluate, inside the registration result.
    :returns: A Nx6 numpy matrix containing the lie algebra representation of the results.
    """
    inv_of_prealignment = np.linalg.inv(prealignment)
    lie_results = np.empty((len(json_data['data']), 6))
    for i, registration in enumerate(json_data['data']):
        m = np.array(registration[key])

        try:
            lie_results[i,:] = se3_log(np.dot(inv_of_prealignment, m))
        except RuntimeError:
            lie_results[i,:] = np.zeros(6)
            eprint('Warning: failed conversion to lie algebra of matrix {}'.format(m))

    return lie_results


def positions_of_registration_data(reg_data, initial_estimates=False, prealignment=np.identity(4)):
    if ('what' in reg_data and
        'trails' == reg_data['what']):
        lie_vectors = lie_tensor_of_trails(reg_data)[(0 if initial_estimates else -1)]
    else:
        key = 'result'
        if initial_estimates:
            key = 'initial_estimate'

        lie_vectors = lie_vectors_of_registrations(reg_data, key=key, prealignment=prealignment)

    return lie_vectors


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

            data_dict['clustering'] = np.ascontiguousarray(cluster_of_points)

        if 'outlier' in registration_data['statistics']:
            outlier_mask = np.zeros(lie_vectors.shape[0], dtype=np.int8)
            for index in registration_data['statistics']['outliers']:
                outlier_mask[index] = 1

            data_dict['outlier'] = np.ascontiguousarray(outlier_mask)

    density = density_of_points(lie_vectors)
    data_dict['density'] = np.ascontiguousarray(density)

    return data_dict


def lie_tensor_of_trails(registration_dataset):
    """
    Arguments
    registration_dataset -- A trails full dataset
    """
    max_trail_length = registration_dataset['metadata']['max_trail_length']
    n_particles = len(registration_dataset['data'])
    positions_of_iterations = np.zeros((max_trail_length, n_particles, 6))

    for i, particle in enumerate(registration_dataset['data']):
        for j in range(max_trail_length):
            latest_position = min(len(particle['trail'])-1, j)
            for k in range(6):
                positions_of_iterations[j, i, k] = particle['trail'][latest_position][k]


    return positions_of_iterations


def points_to_vtk(points, filename, data=None):
    pointsToVTK(filename,
                np.ascontiguousarray(points[:,0]),
                np.ascontiguousarray(points[:,1]),
                np.ascontiguousarray(points[:,2]),
                data = data)

def registration2lie_cli():
    dataset = json.load(sys.stdin)
    json.dump(lie_vectors_of_registrations(dataset).tolist(), sys.stdout)


def center_around_gt_cli():
    dataset = json.load(sys.stdin)
    ground_truth_inv = np.matrix(dataset['metadata']['ground_truth']).I

    for i, result in enumerate(dataset['data']):
        result_matrix = np.matrix(result['result'])
        estimate_matrix = np.matrix(result['initial_estimate'])

        centered_result = ground_truth_inv * result_matrix
        centered_estimate = ground_truth_inv * estimate_matrix

        dataset['data'][i]['result'] = centered_result.tolist()
        dataset['data'][i]['initial_estimate'] = centered_estimate.tolist()

    dataset['metadata']['ground_truth'] = np.identity(4).tolist()

    json.dump(dataset, sys.stdout)
