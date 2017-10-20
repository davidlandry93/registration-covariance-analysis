#!/usr/bin/env python3

import argparse
import json
import numpy as np
import pyevtk
import sys

from util import parse_dims


if __name__ == '__main__':
    json_data = json.load(sys.stdin)

    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='The name of the file where to export the plot')
    parser.add_argument('--dims', type=str, default='0,1,2', help='Comma separated list of the dimensions to extract from the covariance matrix')
    parsed_args = parser.parse_args()

    dims = parse_dims(parsed_args.dims)

    n_iterations = len(json_data['data'][0]['trail'])
    n_particles = len(json_data['data'])
    positions_of_iterations = np.zeros((n_iterations, n_particles, 6))

    for i in range(n_iterations):
        positions_of_iterations[i] = np.zeros((n_particles, 6))
        for j, trail in enumerate(json_data['data']):
            positions_of_iterations[i,j,:] = np.array(trail['trail'])[i]


    for i, particle_positions in enumerate(positions_of_iterations):
        filename = parsed_args.output + '_{0:03d}'.format(i)
        print(filename)

        pyevtk.hl.pointsToVTK(filename,
                              np.ascontiguousarray(particle_positions[:,dims[0]]),
                              np.ascontiguousarray(particle_positions[:,dims[1]]),
                              np.ascontiguousarray(particle_positions[:,dims[2]]),
                              data = None)
