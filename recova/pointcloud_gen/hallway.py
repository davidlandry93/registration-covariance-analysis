#!/usr/bin/env python3

import argparse
import numpy as np
import plyfile

from util import run_subprocess

TEMP_FILENAME = '/tmp/hallway.ply'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('npoints', type=int, help='Number of points comprising the cube')
    parser.add_argument('noise', type=float, help='Standard deviation of the noise of the points in every axis')
    parser.add_argument('size', type=float, help='Size of an edge of the cube', default=1.0)
    parser.add_argument('output', type=str, help='The file where to output the pointcloud')

    args = parser.parse_args()

    pointcloud = np.empty((0,3))

    n_points_per_face = int(args.npoints / 4)
    p = np.zeros((4 * n_points_per_face,), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])

    for face in range(2,6):
        # Generate random points inside the cube.
        points_of_face = np.random.rand(int(args.npoints / 4), 3) * args.size - args.size/2.

        # Fix a dimension so that the point is on a face of the cube.
        points_of_face[:, face // 2] = float(face % 2) * args.size - args.size/2.

        # Add noise.
        points_of_face += np.random.normal(scale=args.noise, size=points_of_face.shape)

        for i, point in enumerate(points_of_face):
            p[i + (face-2) * n_points_per_face] = tuple(point)

    el = plyfile.PlyElement.describe(p, 'vertex', val_types={'x': 'f8', 'y': 'f8', 'z': 'f8'}, )
    plyfile.PlyData([el]).write(TEMP_FILENAME)

    # Convert the ply to pcd
    run_subprocess('pcl_ply2pcd {} {}'.format(TEMP_FILENAME, args.output))
