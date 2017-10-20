
import argparse
import csv
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
import plyfile
import time

from datasets import create_registration_dataset
from registration_algorithm import AlgorithmFactory
from util import run_subprocess


def make_axes_equal(xs, ys, zs, ax):
    """
    Reads the data from xs, ys and zs, and adds bogus points so that the scale of the axes is equal.
    """
    max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max()

    # Make a bounding box so that the axes are equal.
    x_bound = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xs.max()+xs.min())
    y_bound = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ys.max()+ys.min())
    z_bound = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zs.max()+zs.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(x_bound, y_bound, z_bound):
        ax.plot([xb], [yb], [zb], 'w')

def plot_ellipsoid(center, a, b, c, rotation_matrix, ax,  color='b'):
    us = np.linspace(0.0, 2.0 * np.pi, 100)
    vs = np.linspace(0.0, np.pi, 100)

    xs = a * np.outer(np.cos(us), np.sin(vs))
    ys = b * np.outer(np.sin(us), np.sin(vs))
    zs = c * np.outer(np.ones_like(us), np.cos(vs))

    for i in range(len(xs)):
        for j in range(len(xs)):
            [xs[i,j], ys[i,j], zs[i,j]] = np.dot([xs[i,j], ys[i,j], zs[i,j]], rotation_matrix) + center

    ax.plot_wireframe(xs, ys, zs, rstride=4, cstride=4, color=color, alpha=0.2)

def plot_covariance_matrix(position, covariance, ax, color='b'):
    """
    Plot an ellipsoid representing a covariance matrix.
    http://stackoverflow.com/a/14958796
    """
    translation_covariance = covariance[0:3,0:3]
    eig_vals, eig_vecs = np.linalg.eig(translation_covariance)
    scales = np.sqrt(eig_vals)

    plot_ellipsoid(position, scales[0], scales[1], scales[2], eig_vecs, ax, color=color)




def read_pcl_file(path: Path):
    # Convert the pcd to ply
    new_path = path.parent / Path(path.stem + ".ply")
    run_subprocess("pcl_pcd2ply {} {}".format(path, new_path))

    ply = plyfile.PlyData.read(new_path.__str__())
    n_points = len(ply.elements[0].data)

    np_data = np.empty((4, n_points))
    for i in range(n_points):
        np_data[0,i] = ply.elements[0].data[i][0]
        np_data[1,i] = ply.elements[0].data[i][1]
        np_data[2,i] = ply.elements[0].data[i][2]
        np_data[3,i] = 1.0

    return np_data


def read_pcl_files(dataset, reading, reference):
    reading_path = dataset.path_of_cloud(args.reading)
    reference_path = dataset.path_of_cloud(args.reference)

    return read_pcl_file(reading_path), read_pcl_file(reference_path)

def subsample_pointcloud(pointcloud, size=5000):
    subsample_of_pts = np.random.choice(pointcloud.shape[1], size=size)
    return pointcloud[:, subsample_of_pts]

def plot_pointcloud(pointcloud, ax, color='k'):
    ax.scatter(pointcloud[0,:], pointcloud[1,:], pointcloud[2,:], s=0.25, color=color)

def fetch_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("kind", help="Kind of dataset to use. <oru|ethz>", type=str)
    parser.add_argument("dataset", help="Path to the dataset", type=str)
    parser.add_argument("reading", help="Index of the pointcloud to use as reading", type=int)
    parser.add_argument("reference", help="Index of the pointcloud to use as reference", type=int)

    return parser.parse_args()

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    args = fetch_args()

    dataset = create_registration_dataset(args.kind, args.dataset)

    reading, reference = read_pcl_files(dataset, args.reading, args.reference)

    t_odom = dataset.odometry_estimate(args.reading, args.reference)
    print("odometry_estimate")
    print(t_odom)

    t_gt = dataset.ground_truth(args.reading, args.reference)
    print("Ground truth")
    print(t_gt)

    print("Sampling ndt...")
    algo = AlgorithmFactory.create('ndt')
    algo.initial_estimate_covariance = 0.05
    ndt_result = algo.compute_covariance_with_dataset(dataset, args.reading, args.reference)
    print(ndt_result['covariance'])

    print("Sampling icp...")
    algo = AlgorithmFactory.create('icp')
    algo.initial_estimate_covariance = 0.05
    icp_result = algo.compute_covariance_with_dataset(dataset, args.reading, args.reference)
    print(icp_result['covariance'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    reading = subsample_pointcloud(reading, 2000)
    reference = subsample_pointcloud(reference, 2000)

    # Center the plot around the ground truth.
    center = t_gt[0:3,3]
    ax.set_xlim(-5 + center[0], 5 + center[0])
    ax.set_ylim(-5 + center[1], 5 + center[1])
    ax.set_zlim(-5 + center[2], 5 + center[2])

    plot_pointcloud(reference, ax)
    plot_pointcloud(np.dot(t_gt, reading), ax, '0.5')

    plot_covariance_matrix(icp_result['mean'][0:3,3], icp_result['covariance'], ax)
    plot_covariance_matrix(ndt_result['mean'][0:3,3], ndt_result['covariance'], ax, color='r')
    plt.show()

