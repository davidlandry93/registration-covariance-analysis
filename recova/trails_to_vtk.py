#!/usr/bin/env python3

import argparse
import json
import multiprocessing
import numpy as np
import pyevtk
import sys

from recova.registration_dataset import points_to_vtk, lie_tensor_of_trails, data_dict_of_registration_data
from recova.util import parse_dims, empty_to_none

def save_one_frame(points, output, data_dict, i):
    filename = output + '_{0:03d}'.format(i)
    print(filename)
    data = empty_to_none(data_dict)

    points_to_vtk(points[i], filename, data)

def cli():
    json_data = json.load(sys.stdin)

    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='The name of the file where to export the plot')
    parser.add_argument('--dims', type=str, default='0,1,2', help='Comma separated list of the dimensions to extract from the covariance matrix')
    parsed_args = parser.parse_args()

    dims = parse_dims(parsed_args.dims)

    lie_tensor = lie_tensor_of_trails(json_data)
    data_dict = data_dict_of_registration_data(json_data)

    with multiprocessing.Pool() as pool:
        pool.starmap(save_one_frame, [(lie_tensor[:,:,dims], parsed_args.output, data_dict, x) for x in range(len(lie_tensor))])


if __name__ == '__main__':
    cli()
