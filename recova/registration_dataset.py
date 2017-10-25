#!/usr/bin/env python3

import numpy as np

from pyevtk.hl import pointsToVTK
from pylie import se3_log
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


def positions_of_registration_data(reg_data, initial_estimates=False):
    if ('metadata' in reg_data and
        'experiment' in reg_data['metadata'] and
        'trail_batch' == reg_data['metadata']['experiment']):
        lie_vectors = lie_tensor_of_trails(reg_data)[(0 if initial_estimates else -1)]
    else:
        key = 'result'
        if initial_estimates:
            key = 'initial_estimate'

        lie_vectors = lie_vectors_of_registrations(reg_data, key=key)

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

    return data_dict


def lie_tensor_of_trails(registration_dataset):
    n_iterations = len(registration_dataset['data'][0]['trail'])
    n_particles = len(registration_dataset['data'])
    positions_of_iterations = np.zeros((n_iterations, n_particles, 6))

    for i in range(n_iterations):
        positions_of_iterations[i] = np.zeros((n_particles, 6))
        for j, trail in enumerate(registration_dataset['data']):
            positions_of_iterations[i,j,:] = np.array(trail['trail'])[i]

    return positions_of_iterations


def points_to_vtk(points, filename, data=None):
    pointsToVTK(filename,
                np.ascontiguousarray(points[:,0]),
                np.ascontiguousarray(points[:,1]),
                np.ascontiguousarray(points[:,2]),
                data = data)
