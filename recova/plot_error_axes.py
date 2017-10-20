#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import paperplot
import pickle

from datasets import create_registration_dataset
from plot_error_function import TranslationErrorFunction
from registration_algorithm import AlgorithmFactory, run_subprocess
from util import ln_se3

POSITIVE_STRINGS = ['y', 'yes', '1', 't', 'true']
DEFAULT_RESULT_FILE = 'error_all_axes.json'

cmd_string = ('./all_dims -algorithm {} -reading {} -reference {} '
              '-focal \"{}\" -n_samples_per_dim {} -filter_center {} '
              '-algo_config {} -output {} -span {}')

def compute_distribution_of_results(dataset, reading, reference, algo, n_samples=1000, cache=''):
    response = None
    if cache in POSITIVE_STRINGS:
        with open('plot_error_axes_samples.pk', 'rb') as sample_file:
            response = pickle.load(sample_file)
    else:
        algo.n_samples = n_samples
        response = algo.compute_covariance_with_dataset(dataset, reading, reference)

        with open('plot_error_axes_samples.pk', 'wb') as cache_file:
            pickle.dump(response, cache_file)

    return response


def compute_error_around_axes(dataset, reading, reference, algo, temp_file=DEFAULT_RESULT_FILE, cache='', use_gt=True, n_samples_per_dim=50):
    focal = None
    if use_gt:
        focal = dataset.ground_truth(reading, reference)
    else:
        algo.n_samples = 1
        registration = algo.compute_covariance_with_dataset(dataset, reading, reference)
        focal = registration['mean']

    formatted_cmd = cmd_string.format(algo.name,
                                      dataset.path_of_cloud(reading),
                                      dataset.path_of_cloud(reference),
                                      json.dumps(focal.tolist()),
                                      n_samples_per_dim,
                                      dataset.center_filter_size,
                                      algo.config,
                                      temp_file,
                                      0.01)
    print(formatted_cmd)

    result = None
    if cache in POSITIVE_STRINGS:
        with open('plot_error_axes_error_f.pk', 'rb') as pickle_file:
            result = pickle.load(pickle_file)
    else:
        response = run_subprocess(formatted_cmd)
        result = json.loads(response)

        with open('plot_error_axes_error_f.pk', 'wb') as cache_file:
            pickle.dump(result, cache_file)

    return result

def lie_biases(result_sampling):
    biases_algebra = np.empty((len(result_sampling['samples']), 6))
    for i, sample in enumerate(result_sampling['samples']):
        m = np.dot(np.linalg.inv(focal), np.array(sample))
        biases_algebra[i, :] = ln_se3(m)


def plot_error_around_axes(result_error, lie_biases):
    lie_focal = np.array(result_error['metadata']['lie_focal'])
    focal = np.array(result_error['metadata']['focal'])

    fig, subplots = plt.subplots(3,2, sharex=True, sharey=True)
    for dimension in range(0,6):

        current_ax = subplots[dimension % 3, dimension // 3]
        current_ax.set_zorder(10)

        len_of_dim = len(result_error['data'][dimension])

        error_vals_matrix = np.empty((len_of_dim, 2))
        for sample_id in range(0, len_of_dim):
            sample = result_error['data'][dimension][sample_id]

            error_vals_matrix[sample_id, 0] = sample['val_of_dim'] - lie_focal[dimension]
            error_vals_matrix[sample_id, 1] = sample['error']

        error_vals_matrix = error_vals_matrix[error_vals_matrix[:,0].argsort()]

        histogram_ax = current_ax.twinx()

        # Put the histogram ax behind the plot ax.
        current_ax.set_zorder(histogram_ax.get_zorder() + 1)
        current_ax.patch.set_visible(False) # Do not show the canvas

        current_ax.plot(error_vals_matrix[:, 0], error_vals_matrix[:, 1], label=str(dimension), color='black')
        histogram_ax.hist(lie_biases[:,dimension],
                          align='mid',
                          bins=50,
                          color='0.7',
                          normed=False,
                          range=current_ax.get_xlim())

        histogram_ax.set_ylabel('N samples')

    subplots[0,0].set_title('Translation error')
    subplots[0,1].set_title('Rotation error')

    subplots[0,0].set_ylabel('x component')
    subplots[1,0].set_ylabel('y component')
    subplots[2,0].set_ylabel('z component')

    subplots[2,0].set_xlabel('Error (m)')
    subplots[2,1].set_xlabel('Error')

    plt.savefig('fig.pdf')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('kind', help='The kind of dataset user. <oru|ethz>.', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('algo', help='The algorithm to assess', type=str, default="icp")
    parser.add_argument('--cache', help='Wether to use cached results', type=str, default='')
    parser.add_argument('--n_samples_per_dim', help='Number of samples per dim for the error function', type=int, default=100)
    parser.add_argument('--n_samples', help='Number of samples to compute the distribution of results', type=int, default=1000)

    args = parser.parse_args()

    dataset = create_registration_dataset(args.kind, args.dataset)
    algo = AlgorithmFactory.create(args.algo)
    algo.initial_estimate_covariance = 0.5
    algo.initial_estimate_covariance_rot = 0.2

    result_error = compute_error_around_axes(dataset, args.reading, args.reference, algo, cache=args.cache, n_samples_per_dim=args.n_samples_per_dim)
    result_sampling = compute_distribution_of_results(dataset, args.reading, args.reference, algo, n_samples=args.n_samples, cache=args.cache)
    plot_error_around_axes(result_error, lie_biases(result_sampling))
