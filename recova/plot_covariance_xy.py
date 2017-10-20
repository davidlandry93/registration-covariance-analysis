#!/usr/bin/env python3

import argparse
import csv
import matplotlib
import matplotlib.lines as lin
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import pickle

import paperplot
from plot_error_function import TranslationErrorFunction
from datasets import create_registration_dataset
from registration_algorithm import AlgorithmFactory
from test_algos import angle_around_z_of_so3

CACHE_FILE_NAME = 'plot_covariance_xy_cache.json'
POSITIVE_STRINGS = ['y', 'yes', '1', 't', 'true']

class RegistrationDistributionPlot:
    def __init__(self, dataset, algo, reading, reference, n_samples=30):
        self.dataset = dataset
        self.algo = algo
        self.reading = reading
        self.reference = reference
        self.n_samples = n_samples
        self.samples = None
        self.estimates = None
        self.covariance = None
        self.mean = None

    def plot_translation_estimates(self, ax):
        gt = self.dataset.ground_truth(self.reading, self.reference)
        odom = self.dataset.odometry_estimate(self.reading, self.reference)

        xy_of_estimates = np.empty((len(self.estimates), 2))
        for i, t in enumerate(self.estimates):
            xy_of_estimates[i] = t[0:2, 3] - gt[0:2,3]
            ax.scatter(xy_of_estimates[:,0], xy_of_estimates[:,1], s=1, color='0.6', label='Perturbated odometry', zorder=4)

    def plot_translation(self, ax):
        gt = self.dataset.ground_truth(self.reading, self.reference)
        odom = self.dataset.odometry_estimate(self.reading, self.reference)

        xy_of_samples = np.empty((len(self.samples), 2))
        for i, t in enumerate(self.samples):
            xy_of_samples[i] = t[0:2, 3] - gt[0:2,3]

        ax.scatter(xy_of_samples[:,0], xy_of_samples[:,1], s=1, color='black', label='Transforms', zorder=4)
        ax.scatter([0.0], [0.0], marker='v', color='black', label='Ground truth', zorder=4)
        # ax.scatter([odom[0,3] - gt[0,3]], [odom[1,3] - gt[1,3]], marker='^', color='black', label='Odometry estimate', zorder=4)

        plot_2d_cov_matrix(self.covariance[0:2, 0:2], self.mean[0:2,3] - gt[0:2,3], ax=ax)

    def plot_rotation(self, ax, range=None):
        gt = self.dataset.ground_truth(self.reading, self.reference)

        theta_of_samples = np.empty(len(self.samples))
        for i, t in enumerate(self.samples):
            theta_of_samples[i] = angle_around_z_of_so3(t[0:3, 0:3]) - angle_around_z_of_so3(gt[0:3, 0:3])

        ax.hist(theta_of_samples, bins=100, color='0.6', range=range)


def angle_of_rotation(rot_matrix):
    trace = np.matrix.trace(rot_matrix)
    cos_of_theta = (np.matrix.trace(rot_matrix) - 1.) / 2.
    return np.arccos(cos_of_theta)

def plot_2d_cov_matrix(cov_matrix, center, ax):
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    angle = np.rad2deg(np.arctan2(eigvecs[1,0], eigvecs[0,0]))

    print(angle)

    e = pat.Ellipse(center, 2*np.sqrt(eigvals[0]), 2*np.sqrt(eigvals[1]), angle)
    ax.add_artist(e)
    e.set_alpha(1.0)
    e.set_fill(False)
    e.set_edgecolor('black')
    e.set_clip_box(ax.bbox)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('kind', help='The kind of dataset user. <oru|ethz>.', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('algo', help='The algorithm to assess', type=str, default="icp")
    parser.add_argument('--n_samples', help='Number of samples to take', type=int, default=30)
    parser.add_argument('--est_pert_var', help='Variance of the perturbation to apply to the initial estimate', type=float, default=0.01)
    parser.add_argument('--cache', help='Whether to use cached values or not', type=str, default='')
    parser.add_argument('--odom', help='Show the odometry or not on the plot', type=str, default='t')
    parser.add_argument('--error_f', help='Also plot the error function under the samples', type=str, default='f')

    args = parser.parse_args()

    paperplot.setup()

    dataset = create_registration_dataset(args.kind, args.dataset)
    algo = AlgorithmFactory.create(args.algo)
    algo.n_samples = args.n_samples
    algo.initial_estimate_covariance = args.est_pert_var
    algo.outlier_ratio = 0.75

    dist = RegistrationDistributionPlot(dataset, algo, args.reading, args.reference)

    results = None
    if args.cache.lower() in POSITIVE_STRINGS:
        with open(CACHE_FILE_NAME, 'rb') as pickle_file:
            results = pickle.load(pickle_file)
    else:
        results = algo.compute_covariance_with_dataset(dataset, args.reading, args.reference)
        with open(CACHE_FILE_NAME, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

    dist.samples = results['samples']
    dist.estimates = results['estimates']
    dist.covariance = results['covariance']
    dist.mean = results['mean']


    fig = paperplot.paper_figure(396, 198)
    ax_tr = fig.add_subplot(1,2,1)
    ax_rot = fig.add_subplot(1,2,2)
    # fig.suptitle('Distribution of the results of registration')

    dist.plot_translation(ax_tr)

    ax_tr.axis('equal')
    ax_tr.set_xlabel('Distance from GT x axis (m)')
    ax_tr.set_ylabel('Distance from GT y axis (m)')
    ax_tr.legend()

    dist.plot_rotation(ax_rot)

    ax_rot.set_ylabel('Number of samples')
    ax_rot.yaxis.tick_right()
    ax_rot.yaxis.set_label_position('right')
    ax_rot.set_xlabel('Rotation error around z axis (rad)')
    ax_rot.legend()

    plt.savefig('fig.pdf')
    plt.show()
