#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

from datasets import create_registration_dataset
from registration_algorithm import AlgorithmFactory
from util import run_subprocess

CACHE_FILE_NAME = 'trail_plot_cache'
POSITIVE_STRINGS = ['y', 'yes', '1', 't', 'true']

class RegistrationTrail:
    def __init__(self, dataset, algo, reading, reference):
        self.dataset = dataset
        self.algo = algo
        self.reading = reading
        self.reference = reference
        self.trail_data = None

    def compute(self):
        cmd_string = ('./trail -algorithm {} -reading {} -reference {} '
                      '-estimate \"{}\" -filter_center {} -algo_config {}').format(
                          self.algo.name,
                          self.dataset.path_of_cloud(self.reading),
                          self.dataset.path_of_cloud(self.reference),
                          json.dumps(self.dataset.odometry_estimate(self.reading, self.reference).tolist()),
                          self.dataset.center_filter_size,
                          self.algo.config)

        print(cmd_string)
        response = run_subprocess(cmd_string)
        print(response)

        self.trail_data = json.loads(response)

        with open(CACHE_FILE_NAME, 'wb') as pickle_file:
            pickle.dump(self.trail_data, pickle_file)


    def load(self):
        with open(CACHE_FILE_NAME, 'rb') as pickle_file:
            self.trail_data = pickle.loads(pickle_file.read())


    def plot(self, ax):
        trail_tfs = np.array(self.trail_data['trail'])
        trail_xy = trail_tfs[:,0:2,3]
        trail_xy = trail_xy - self.dataset.ground_truth(self.reading, self.reference)[0:2,3]

        n_iter = trail_xy.shape[0]

        ax.plot(trail_xy[:,0], trail_xy[:,1], marker='', label='Trail of optimization', color='black')
        # ax.scatter([trail_xy[0,0]], [trail_xy[0,1]], marker='o', label='Start', color='black', zorder=5)
        ax.scatter([trail_xy[n_iter - 1,0]], [trail_xy[n_iter - 1,1]], marker='p', label='After {} iterations'.format(n_iter), color='black', zorder=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', help='The kind of dataset user. <oru|ethz>.', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('algo', help='The algorithm to assess', type=str, default="icp")
    parser.add_argument('--cache', help='Use cache instead of computing data', type=str, default="false")
    args = parser.parse_args()

    dataset = create_registration_dataset(args.kind, args.dataset)
    algo = AlgorithmFactory.create(args.algo)

    trail = RegistrationTrail(dataset, algo, args.reading, args.reference)

    if args.cache in POSITIVE_STRINGS:
        trail.load()
    else:
        trail.compute()

    fig, ax = plt.subplots()
    ax.axis('equal')
    trail.plot(ax)
    plt.show()

