#!/usr/bin/env python3

import argparse
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np

from datasets import create_registration_dataset
from plot_error_function import TranslationErrorFunction
from registration_algorithm import AlgorithmFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('kind', help='The kind of dataset user. <oru|ethz>.', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('algo', help='The algorithm to assess', type=str, default="icp")
    parser.add_argument('--span_t', help='The span to plot in translation', type=float, default=1.0)
    parser.add_argument('--span_r', help='The span of the data to plot in rotation', type=float, default=0.1)
    parser.add_argument('--n_samples_t', help='The number of samples to collect for translation data', type=int, default=50)
    parser.add_argument('--n_samples_r', help='The number of samples to collect for rotation data', type=int, default=10)
    args = parser.parse_args()

    dataset = create_registration_dataset(args.kind, args.dataset)
    algo = AlgorithmFactory.create(args.algo)
    algo.n_samples = 10
    algo.initial_estimate_covariance = 0.03


    print('Computing covariance of registration pair...')
    results = algo.compute_covariance_with_dataset(dataset, args.reading, args.reference)
    print('Done')

    mean = np.array(results['mean'])
    covariance = np.array(results['covariance'])
    projected_covariance = covariance[0:2, 0:2]
    print(projected_covariance)

    eigvals, eigvecs = np.linalg.eig(projected_covariance)

    print(eigvals)
    print(eigvecs)

    print('Sampling objective function of registration pair...')
    error_f = TranslationErrorFunction(dataset,
                                       algo,
                                       args.reading,
                                       args.reference,
                                       args.n_samples_t,
                                       args.n_samples_r)

    error_f.compute(args.span_t, args.span_r, at=mean)
    error_f.read_data()
    error_f.plot()

    e = pat.Ellipse([mean[0,3], mean[1,3]], 2*np.sqrt(eigvals[0]), 2*np.sqrt(eigvals[1]), np.arctan2(eigvecs[0,1], eigvecs[0,0]))
    plt.gca().add_artist(e)
    e.set_alpha(1.0)
    e.set_fill(False)
    e.set_edgecolor('black')
    e.set_clip_box(plt.gca().bbox)

    plt.plot([mean[0,3]], [mean[1,3]], color='black', marker='o')

    gt = dataset.ground_truth(args.reading, args.reference)
    plt.plot([gt[0,3]], [gt[1,3]], color='black', marker='x')

    odom = dataset.odometry_estimate(args.reading, args.reference)
    plt.plot([odom[0,3]], [odom[1,3]], color='black', marker='^')

    plt.axis('equal')
    plt.show()
