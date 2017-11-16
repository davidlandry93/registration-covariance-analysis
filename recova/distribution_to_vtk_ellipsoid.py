#!/usr/bin/env python3
"""
Create a vtu file representing a gaussian distribution in 3D.

Accept a json dictionary with two keys, mean and covariance.

"""

import argparse
import json
import numpy as np
import pyevtk
import sys

from pylie import se3_log
from recova.util import parse_dims


def make_ellipsoid_mesh(a,b,c, resolution_u = 10, resolution_v = 10):
    u = np.linspace(0, 2*np.pi, resolution_u, endpoint=True)
    v = np.linspace(0, np.pi, resolution_v, endpoint=True)

    U,V = np.meshgrid(u,v, indexing='ij')

    X = a * np.cos(U) * np.sin(V)
    Y = b * np.sin(U) * np.sin(V)
    Z = c * np.cos(V)

    # Flatten the list of points.
    points = []
    for i in range(resolution_u):
        for j in range(resolution_v):
            point = [X[i,j], Y[i,j], Z[i,j]]
            points.append(point)

    # Create the connectivity.
    # Every point is connected to its neighbours on the meshgrid.
    connectivity = []
    offsets = []
    for i in range(resolution_u - 1):
        for j in range(resolution_v - 1):
            point_id = i*resolution_v + j

            element_to_add = [point_id, point_id + resolution_v, point_id + resolution_v + 1, point_id + 1]
            connectivity.extend(element_to_add)

            if not offsets:
                offsets = [len(element_to_add)]
            else:
                offsets.append(offsets[-1] + len(element_to_add))

    return np.array(points), np.array(connectivity), np.array(offsets)


def save_evtk_unstructured_grid(filename, points, connectivity, offsets):
    cell_types = np.array([pyevtk.vtk.VtkQuad.tid] * len(offsets))
    pyevtk.hl.unstructuredGridToVTK(filename,
                                    np.ascontiguousarray(points[:,0]),
                                    np.ascontiguousarray(points[:,1]),
                                    np.ascontiguousarray(points[:,2]),
                                    np.ascontiguousarray(connectivity),
                                    np.ascontiguousarray(offsets),
                                    np.ascontiguousarray(cell_types))


def apply_t_to_points(points, T):
    homogeneous_points = np.ones((4,len(points)))
    homogeneous_points[0:3,:] = points.T

    transformed_points = np.dot(T, homogeneous_points)

    return transformed_points[0:3, :].T


def distribution_to_vtk_ellipsoid(mean, covariance, filename):
    eig_vals, eig_vecs = np.linalg.eig(covariance)

    T = np.identity(4)
    T[0:3,0:3] = eig_vecs
    T[0:3,3] = mean

    print('T of ellipsoid: {}'.format(T))
    print('Eigvals: {}'.format(eig_vals))

    # Replace negative eigenvalues by a very small number.
    for i in range(len(eig_vals)):
        if eig_vals[i] < 0.0:
            eig_vals[i] = 1e-6

    points, connectivity, offsets = make_ellipsoid_mesh(*np.sqrt(eig_vals))
    print(filename)

    points_transformed = apply_t_to_points(points, T)

    save_evtk_unstructured_grid(filename, points_transformed, connectivity, offsets)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='The name of the file where to export the plot')
    parser.add_argument('--dims', type=str, default='0,1,2', help='Comma separated list of the dimensions to extract from the covariance matrix')
    parsed_args = parser.parse_args()

    input_dict = json.load(sys.stdin)
    mean = np.array(input_dict['mean'])
    covariance = np.array(input_dict['covariance'])

    mean_lie = se3_log(mean)

    dims = parse_dims(parsed_args.dims)

    # Extract the appropriate dims from the covariance matrix.
    covariance = covariance[dims][:,dims]

    distribution_to_vtk_ellipsoid(mean_lie[dims], covariance, parsed_args.output)


if __name__ == '__main__':
    cli()
