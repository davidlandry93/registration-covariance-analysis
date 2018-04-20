import argparse
import json
import multiprocessing
import numpy as np
import pyevtk
import sys

from recova.registration_dataset import points_to_vtk, lie_tensor_of_trails, data_dict_of_registration_data
from recova.util import parse_dims, empty_to_none

def save_one_frame(points, output, data_dict, i, pre_transform):
    filename = output + '_{0:03d}'.format(i)
    print(filename)
    data = empty_to_none(data_dict)

    points_to_vtk(points[i], filename, data)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='The name of the file where to export the plot')
    parser.add_argument('--dims', type=str, default='0,1,2', help='Comma separated list of the dimensions to extract from the covariance matrix')
    parser.add_argument('--center_around_gt', action='store_true', help='Center the points around the ground truth solution')
    parsed_args = parser.parse_args()

    dims = parse_dims(parsed_args.dims)

    json_data = json.load(sys.stdin)

    lie_tensor = lie_tensor_of_trails(json_data)
    data_dict = data_dict_of_registration_data(json_data)

    if parsed_args.center_around_gt:
        print('Applying pre transform')
        pre_transform = np.linalg.inv(np.array(json_data['metadata']['ground_truth']))
    else:
        print('Not applying pre transform')
        pre_transform = np.identity(4)

    with multiprocessing.Pool() as pool:
        pool.starmap(save_one_frame, [(lie_tensor[:,:,dims], parsed_args.output, data_dict, x, pre_transform) for x in range(len(lie_tensor))])


if __name__ == '__main__':
    cli()
