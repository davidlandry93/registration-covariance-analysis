#!/usr/bin/env python3

import argparse
import math
import numpy as np
import plyfile

from util import run_subprocess

TEMP_FILENAME = '/tmp/cylinder.ply'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('npoints', type=int, help='Number of points comprising the cube')
    parser.add_argument('noise', type=float, help='Standard deviation of the noise of the points in every axis')
    parser.add_argument('output', type=str, help='The file where to output the pointcloud')
    parser.add_argument('height', type=float, help='Height of the cylinder')
    parser.add_argument('radius', type=float, help='Radius of the base')

    args = parser.parse_args()

    # Compute the proportion of points on each face
    area_of_base = 2 * math.pi * args.radius * args.radius
    area_of_side = 2 * math.pi * args.radius * args.height
    total_area = 2*area_of_base + area_of_side
    n_points_on_side = int(args.npoints * area_of_side / total_area)
    n_points_on_base = int(args.npoints * area_of_base / total_area)

    n_points = 2 * n_points_on_base + n_points_on_side

    # Generate points for bases
    numpy_points = np.zeros((0,3))
    for base in range(0,2):
        parameters_of_points = np.random.rand(n_points_on_base, 2)

        # See http://mathworld.wolfram.com/DiskPointPicking.html for correct sampling of circle.
        parameters_of_points[:,0] = args.radius * np.sqrt(parameters_of_points[:,0])
        parameters_of_points[:,1] *= 2*math.pi

        points_of_base = np.empty((n_points_on_base, 3))
        points_of_base[:,0] = np.cos(parameters_of_points[:,1]) * parameters_of_points[:,0]
        points_of_base[:,1] = np.sin(parameters_of_points[:,1]) * parameters_of_points[:,0]
        points_of_base[:,2] = args.height * float(base) - args.height / 2.

        numpy_points = np.vstack([numpy_points, points_of_base])

    # Generate points for the side
    parameters_of_points = np.random.rand(n_points_on_side, 2)
    parameters_of_points[:,0] *= args.height
    parameters_of_points[:,1] *= 2 * math.pi

    points_of_side = np.empty((n_points_on_side, 3))
    points_of_side[:,0] = args.radius * np.cos(parameters_of_points[:,1])
    points_of_side[:,1] = args.radius * np.sin(parameters_of_points[:,1])
    points_of_side[:,2] = parameters_of_points[:,0] - args.height / 2.

    numpy_points = np.vstack([points_of_side, numpy_points])

    # Add noise to points
    numpy_points += np.random.normal(scale=args.noise, size=numpy_points.shape)
    print(numpy_points)

    # Make a ply file from the points
    pointcloud = np.zeros((n_points,), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
    for i, point in enumerate(numpy_points):
        pointcloud[i] = tuple(point)

    print(pointcloud)

    el = plyfile.PlyElement.describe(pointcloud, 'vertex', val_types={'x': 'f8', 'y': 'f8', 'z': 'f8'}, )
    print(el)
    plyfile.PlyData([el]).write(TEMP_FILENAME)

    # Convert the ply to pcd
    run_subprocess('pcl_ply2pcd {} {}'.format(TEMP_FILENAME, args.output))
