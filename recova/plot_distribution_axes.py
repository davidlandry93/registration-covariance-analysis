#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

from util import ln_se3

def plot_distributions_on_axes(lie_biases):
    fig, subplots = plt.subplots(3,2, sharex=False, sharey=True)

    for dimension in range(0,6):
        current_ax = subplots[dimension % 3, dimension // 3]

        current_ax.hist(lie_biases[:,dimension],
                        bins=80,
                        color='0.4',
                        log=True)

    for row in range(0,3):
        subplots[row,0].set_ylabel('n samples')

    subplots[2,0].set_xlabel('GT Error (m)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='Json file containing the results to plot')

    args = parser.parse_args()

    with open(args.input) as datafile:
        data = json.load(datafile)

        ground_truth = np.array(data['metadata']['ground_truth'])

        lie_biases = np.empty((len(data['data']), 6))
        for i, registration_run in enumerate(data['data']):
            result = np.array(registration_run['result'])
            bias = np.dot(np.linalg.inv(ground_truth), result)
            bias_lie = ln_se3(bias)

            lie_biases[i, :] = bias_lie

        print(lie_biases)
        print(np.average(lie_biases, axis=0))
        print(np.std(lie_biases, axis=0))

        plot_distributions_on_axes(lie_biases)
        plt.show()
