import argparse
import numpy as np

from recov.pointcloud_io import pointcloud_to_vtk, pointcloud_to_pcd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('--n-points', type=int, default=100000)
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=0.01)
    args = parser.parse_args()

    parameters = np.random.rand(args.n_points, 2)
    parameters[:,0] = args.radius * np.sqrt(parameters[:,0])
    parameters[:,1] *= 2*np.pi

    points = np.empty((args.n_points, 3))
    points[:,0] = np.cos(parameters[:,1]) * parameters[:,0]
    points[:,1] = np.sin(parameters[:,1]) * parameters[:,0]
    points[:,2] = 0.0

    points += np.random.normal(loc=0.0, scale=args.noise, size=(args.n_points, 3))

    pointcloud_to_pcd(points, args.output)
