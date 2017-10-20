#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import paperplot

from datasets import create_registration_dataset
from plot_error_function import TranslationErrorFunction
from plot_objective_function import center_point
from plot_trail import RegistrationTrail
from registration_algorithm import AlgorithmFactory

def compute_span(ax):
    xbounds = ax.get_xbound()
    ybounds = ax.get_ybound()

    return max(abs(xbounds[1] - xbounds[0]), abs(ybounds[1] - ybounds[0])) * 1.5

def plot_trail_over_error(dataset, reading, reference, algo):
    fig, ax = plt.subplots()
    ax.axis('equal')

    trail = RegistrationTrail(dataset, algo, reading, reference)
    trail.compute()
    trail.plot(ax)

    gt = dataset.ground_truth(reading, reference)
    center = center_point(ax.get_xbound(), ax.get_ybound())
    gt[0,3] += center[0]
    gt[1,3] += center[1]

    error_f = TranslationErrorFunction(dataset, algo, reading, reference, 200)
    error_f.compute(compute_span(ax), 0.1, at=gt)
    error_f.read_data()
    error_f.plot(fig, ax)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('kind', help='The kind of dataset user. <oru|ethz>.', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('algo', help='The algorithm to assess', type=str, default="icp")

    args = parser.parse_args()

    dataset = create_registration_dataset(args.kind, args.dataset)
    algo = AlgorithmFactory.create(args.algo)

    plot_trail_over_error(dataset, args.reading, args.reference, algo)
