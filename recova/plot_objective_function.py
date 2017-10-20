#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import paperplot
import pickle

from datasets import create_registration_dataset
from plot_error_function import TranslationErrorFunction
from plot_trail import RegistrationTrail
from registration_algorithm import AlgorithmFactory
from test_algos import angle_around_z_of_so3, angle_of_so3

from registration_algorithm import IcpErrorMinimizer, NdtMatcherType

CACHE_FILE_NAME = 'objective_function_plot_cache'
POSITIVE_STRINGS = ['y', 'yes', '1', 't', 'true']

def center_point(xbounds, ybounds):
    return ((xbounds[1] - xbounds[0]) / 2 + xbounds[0], (ybounds[1] - ybounds[0]) / 2 + ybounds[0])


def plot_covariance_ellipse(ax, data):
    cov_matrix = data['covariance'][0:2,0:2]
    center = data['mean'][0:2,3]
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    print('cov-matrix')
    print(cov_matrix)

    angle = np.rad2deg(np.arctan2(eigvecs[1,0], eigvecs[0,0]))

    e = pat.Ellipse(center, 2*np.sqrt(eigvals[0]), 2*np.sqrt(eigvals[1]), angle)
    ax.add_artist(e)
    e.set_alpha(1.0)
    e.set_fill(False)
    e.set_edgecolor('black')
    e.set_clip_box(ax.bbox)



def compute_span(xbounds, ybounds):
    return max(abs(xbounds[1] - xbounds[0]), abs(ybounds[1] - ybounds[0])) * 1.5


def compute_matches_data(algo, dataset, reading, reference):
    data = algo.compute_covariance_with_dataset(dataset, reading, reference)

    with open(CACHE_FILE_NAME, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

    return data
def load_matches_data():
    results = None
    with open(CACHE_FILE_NAME, 'rb') as pickle_file:
        results = pickle.load(pickle_file)

    return results


def compute_objective_function_data(f, algo, dataset, reading, reference, span, center=None):

    gt = dataset.ground_truth(reading, reference)
    if center is not None:
        gt[0, 3] = center[0]
        gt[1, 3] = center[1]

    f.compute(span, 1, at=gt)


def plot_matches(ax, data, ground_truth):
    xy_of_samples = np.empty((len(data['samples']), 2))
    for i, t in enumerate(data['samples']):
        xy_of_samples[i] = t[0:2, 3] - ground_truth[0:2, 3]

    ax.scatter(xy_of_samples[:,0], xy_of_samples[:,1], s=1, color='black', label='Registration results', zorder=4)

def plot_odom(ax, data, ground_truth):
    xy_of_estimates = np.empty((len(data['estimates']), 2))
    for i, t in enumerate(data['estimates']):
        xy_of_estimates[i] = t[0:2, 3] - ground_truth[0:2, 3]

    ax.scatter(xy_of_estimates[:,0], xy_of_estimates[:,1], s=1, color='0.7', label='Perturbated odometry', zorder=4)

def plot_ground_truth(ax, dataset, reading, reference):
    gt = dataset.ground_truth(reading, reference)
    ax.scatter([[0.0]], [[0.0]], color='black', marker='v', zorder=5, label='Ground truth')

def objective_function_plot(kind, dataset_path, reading, reference, algo_name, cache=False, span=0.0):
    paperplot.setup()

    dataset = create_registration_dataset(kind, pathlib.Path(dataset_path))
    algo = AlgorithmFactory.create(algo_name)
    algo.n_samples = 40
    algo.initial_estimate_covariance = 0.01
    if algo_name == 'icp':
        algo.error_minimizer = IcpErrorMinimizer.POINT_TO_PLANE

    trail = RegistrationTrail(dataset, algo, reading, reference)

    matches = None
    if not cache:
        matches = compute_matches_data(algo, dataset, reading, reference)
    else:
        matches = load_matches_data()

    gt = dataset.ground_truth(reading, reference)
    t_mean_gt = np.dot(np.linalg.inv(matches['mean']), gt)
    print(t_mean_gt)
    angle = angle_around_z_of_so3(t_mean_gt[0:3,0:3])
    print('ANGULAR ERROR Z {}'.format(angle))
    angle = angle_of_so3(t_mean_gt[0:3,0:3])
    print('ANGLE ERROR {}'.format(angle))

    print('T form mean to gt')
    print(t_mean_gt)

    if not cache:
        trail.compute()
    else:
        trail.load()

    fig = paperplot.paper_figure(396,300)
    ax = fig.add_subplot(1,1,1)
    ax.axis('equal')


    plot_ground_truth(ax, dataset, reading, reference)
    plot_matches(ax, matches, gt)
    plot_odom(ax, matches, gt)
    trail.plot(ax)

    # We have to reproject the data in the transform space instead of the ground truth space
    center = center_point(ax.get_xbound(), ax.get_ybound()) + gt[0:2,3]

    if span == 0.:
        span = compute_span(ax.get_xbound(), ax.get_ybound())

    f = TranslationErrorFunction(dataset, algo, reading, reference, 200, 1)

    if not cache:
        compute_objective_function_data(f, algo, dataset, reading, reference, span, center)

    f.read_data()
    f.plot(fig, ax)

    plot_covariance_ellipse(ax, matches)
    ax.legend()
    ax.set_xlabel('Distance from ground truth x axis (m)')
    ax.set_ylabel('Distance from ground truth y axis (m)')


    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('kind', help='The kind of dataset user. <oru|ethz>.', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('algo', help='The algorithm to assess', type=str, default="icp")
    parser.add_argument('--cache', help='Use the results in cache', type=str, default="false")
    parser.add_argument('--span', help='Manually define the span, in meters', type=float, default=0.0)

    args = parser.parse_args()

    cache = args.cache.lower() in POSITIVE_STRINGS

    objective_function_plot(args.kind, args.dataset, args.reading, args.reference, args.algo, cache, args.span)

