#!/usr/bin/env python3

import argparse
import subprocess
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx
import numpy as np
import pathlib
import paperplot
from multiprocessing import Pool
import json
from math import sin, cos

from datasets import create_registration_dataset
from registration_algorithm import AlgorithmFactory
from util import ln_se3


POSITIVE_STRINGS = ['y', 'yes', '1', 't', 'true']


def compute_matches_data(algo, dataset, reading, reference, odometry_estimate = None):
    data = algo.compute_covariance_with_dataset(dataset, reading, reference, odometry_estimate)

    return data

def worker_entry_function(dataset, pair):
    reading, reference = pair
    print("===Launching test for reading {} and ref {}".format(reading, reference))

    gt = dataset.ground_truth(reading, reference)

    algo = AlgorithmFactory.create('icp')
    algo.n_samples = 7

    ratios = np.arange(1e-5, 1.0, 0.05)
    res = []
    for ratio in ratios:
        algo.trimmed_dist_filter_ratio = float(ratio)
        algo.initial_estimate_covariance = 0.1
        algo.initial_estimate_covariance_rot = 0.05

        try:
            matches = compute_matches_data(algo, dataset, reading, reference)
        except subprocess.CalledProcessError:
            print("Fail to match")
            res.append(np.inf)
            continue

        t_mean_gt = np.dot(matches['mean'], np.linalg.inv(gt))
        bias = ln_se3(t_mean_gt)
        norm_bias = np.linalg.norm(bias)
        print(ratio, "norm_bias", norm_bias, "bias", bias)
        res.append(norm_bias)

    return (ratios*100).tolist(), res


def outlier_bias_plot(kind, dataset, first_pts, last_pts, cache=False, span=0.0, nb_workers=4):
    paperplot.setup()

    dataset_name = dataset.path_to_dataset.stem
    output_filename = "result_outlier_bias_{}_{}-{}.json".format(dataset_name, first_pts, last_pts)

    pairs_of_scan = [(i, i + 1) for i in range(first_pts, last_pts)]

    if not cache:
        partial_callback = partial(worker_entry_function, dataset)

        p = Pool(nb_workers)
        results = p.map(partial_callback, pairs_of_scan)
    else:
        with open(output_filename, 'r') as jsonfile:
            results = json.load(jsonfile)

    print(results)

    # Export to json the result
    with open(output_filename, 'w') as jsonfile:
        json.dump(results, jsonfile)

    # Plot result
    fig = paperplot.paper_figure(396, 300)
    ax = fig.add_subplot(1, 1, 1)

    # Set colormap
    cm = plt.get_cmap('gist_heat')
    scalar_map = cmx.ScalarMappable(norm=colors.Normalize(vmin=first_pts, vmax=last_pts*1.1), cmap=cm)

    for (ref, read), (ratios, res) in zip(pairs_of_scan, results):
        label = "Pair {}-{}".format(ref, read)
        color_val = scalar_map.to_rgba(read)
        ax.plot(ratios, res, label=label, color=color_val, linewidth=2.0)
    ax.legend()
    ax.set_title("Dataset ${}$".format(dataset_name))
    ax.set_xlabel('Ratio of inlier(\%)')
    ax.set_ylabel('$\|Bias\|$ (m)')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('kind', help='The kind of dataset user. <oru|ethz>.', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('first', help='Index of the first reference pointcloud to use', type=int)
    parser.add_argument('last', help='Index of the last reference pointcloud to use', type=int)
    parser.add_argument('--cache', help='Use the results in cache', type=str, default="false")
    parser.add_argument('--span', help='Manually define the span, in meters', type=float, default=0.0)
    parser.add_argument('--nb_workers', help='Number of worker in the worker pool', type=int, default=2)

    args = parser.parse_args()

    cache = args.cache.lower() in POSITIVE_STRINGS

    dataset = create_registration_dataset(args.kind, pathlib.Path(args.dataset))

    outlier_bias_plot(args.kind, dataset, args.first, args.last, cache, args.span, args.nb_workers)

