#!/usr/bin/env python3

import argparse
import numpy as np
import plyfile

def points_of_cube_face(face_index, size, n_points):
    # Generate a collection of random points.
    points_of_face = np.random.rand(int(n_points / 6), 3) * size - size/2.

    # Modify one of the dimensions of the point so that it lies on a face of the cube.
    # If the face is even, place it at the bottom of the cube. Otherwise place it on top.
    points_of_face[:, face_index // 2] = (float(face_index % 2) * size - size / 2.)

    return points_of_face

def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('npoints', type=int, help='Number of points comprising the cube')
    parser.add_argument('noise', type=float, help='Standard deviation of the noise of the points in every axis')
    parser.add_argument('size', type=float, help='The size of one edge', default=1.0)
    parser.add_argument('output', type=str, help='The file where to output the pointcloud')

    args = parser.parse_args()

    pointcloud = np.empty((0,3))

    n_points_per_face = int(args.npoints / 6)

    p = np.zeros((6 * n_points_per_face,), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])

    for face in range(0,6):
        points = points_of_cube_face(face, args.size, args.npoints)
        points += np.random.normal(scale=args.noise, size=points.shape)

        for i, point in enumerate(points):
            p[i + face * n_points_per_face] = tuple(point)

    el = plyfile.PlyElement.describe(p, 'vertex', val_types={'x': 'f8', 'y': 'f8', 'z': 'f8'}, )
    plyfile.PlyData([el]).write(args.output)


if __name__ == '__main__':
    cli()
