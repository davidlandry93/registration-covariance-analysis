#!/usr/bin/env python3


import argparse
import json
import numpy as np
import sys

from recova.clustering_dbscan import lie_vectors_of_registrations
from pyevtk.hl import pointsToVTK
from pylie import se3_log
from recova.registration_dataset import points_to_vtk, data_dict_of_registration_data, positions_of_registration_data
from recova.util import POSITIVE_STRINGS, empty_to_none


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Where to save the vtk file.')
    parser.add_argument('--dim1', type=int, default=0)
    parser.add_argument('--dim2', type=int, default=1)
    parser.add_argument('--dim3', type=int, default=2)
    parser.add_argument('--rotation', action='store_true', help='Print rotation dimensions instead of translation.')
    parser.add_argument('--initial-estimate', type=str, default='false', help='Print initial estimates instead of results.')
    args = parser.parse_args()

    json_data = json.load(sys.stdin)

    if args.rotation:
        dims = (3,4,5)
    else:
        dims = (args.dim1, args.dim2, args.dim3)

    points = positions_of_registration_data(json_data, (args.initial_estimate in POSITIVE_STRINGS))

    data_dict = empty_to_none(data_dict_of_registration_data(json_data))
    points_to_vtk(points[:, dims], args.output, data_dict)


if __name__ == '__main__':
    cli()
