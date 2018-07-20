#!/usr/bin/env python3


import argparse
import json
import numpy as np
import sys

from recova.registration_dataset import lie_vectors_of_registrations
from pyevtk.hl import pointsToVTK
from recova.registration_dataset import data_dict_of_registration_data, positions_of_registration_data
from recova.util import POSITIVE_STRINGS, empty_to_none, rotation_around_z_matrix
from recov.pointcloud_io import pointcloud_to_vtk


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Where to save the vtk file.')
    parser.add_argument('--dim1', type=int, default=0)
    parser.add_argument('--dim2', type=int, default=1)
    parser.add_argument('--dim3', type=int, default=2)
    parser.add_argument('--rotation', action='store_true', help='Print rotation dimensions instead of translation.')
    parser.add_argument('--initial-estimate', action='store_true', help='Print initial estimates instead of results.')
    parser.add_argument('--center_around_gt', action='store_true', help='Center the results around the ground truth')
    parser.add_argument('-rz', '--rotation_around_z', type=float, default=0.0)
    args = parser.parse_args()

    json_data = json.load(sys.stdin)

    if args.rotation:
        dims = (3,4,5)
    else:
        dims = (args.dim1, args.dim2, args.dim3)

    left_multiply = rotation_around_z_matrix(args.rotation_around_z)
    right_multiply = np.linalg.inv(rotation_around_z_matrix(args.rotation_around_z))

    if args.center_around_gt:
        left_multiply = np.dot(left_multiply, np.linalg.inv(np.array(json_data['metadata']['ground_truth'])))

    points = positions_of_registration_data(json_data, (args.initial_estimate in POSITIVE_STRINGS), left_multiply=left_multiply, right_multiply=right_multiply)

    print(points)

    data_dict = empty_to_none(data_dict_of_registration_data(json_data))
    pointcloud_to_vtk(points[:, dims], args.output, data_dict)


if __name__ == '__main__':
    cli()
