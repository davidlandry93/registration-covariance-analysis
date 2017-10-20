#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import paperplot

from datasets import create_registration_dataset
from plot_covariance_xy import RegistrationDistributionPlot
from plot_error_function import TranslationErrorFunction
from plot_trail_over_error import compute_span
from registration_algorithm import AlgorithmFactory

POSITIVE_STRINGS = ['y', 'yes', '1', 't', 'true']

KIND = 'ethz'
DATASET = 'eth/wood_autumn'
READING = 11
REFERENCE = 10

ROTATION_SPAN = [-0.01, 0.01]


if __name__ == '__main__':
    cache_file = {'ICP': 'fig_dist_over_error_icp',
                  'NDT': 'fig_dist_over_error_ndt'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', help='Used cached data instead of computing new data', type=str, default='f')
    args = parser.parse_args()

    paperplot.setup()
    fig = paperplot.paper_figure(396, 396)
    ax_icp = fig.add_subplot(2,2,1)
    ax_rot_icp = fig.add_subplot(2,2,3)
    ax = {'ICP': ax_icp,
          'NDT': fig.add_subplot(2,2,2, sharex=ax_icp, sharey=ax_icp)}

    ax['ICP'].axis('equal')
    ax['NDT'].axis('equal')

    ax_icp_rot = fig.add_subplot(2,2,3)
    ax_icp_rot.set_xlim(ROTATION_SPAN)

    ax_rot = {}
    ax_rot['ICP'] = ax_icp_rot
    ax_rot['NDT'] = fig.add_subplot(2,2,4, sharex=ax_icp_rot)

    dataset = create_registration_dataset(KIND, DATASET)

    algo = {}
    for algo_name in ['ICP', 'NDT']:
        algo[algo_name] = AlgorithmFactory.create(algo_name)
        algo[algo_name].initial_estimate_covariance = 0.01
        algo[algo_name].n_samples = 100
        dist = RegistrationDistributionPlot(dataset, algo[algo_name], READING, REFERENCE)

        if args.cache.lower() in POSITIVE_STRINGS:
            dist.load(cache_file[algo_name])
        else:
            dist.compute(cache_file[algo_name])

        dist.plot_translation(ax[algo_name])

        ax[algo_name].set_xlabel('Distance from GT x axis (m)')
        ax[algo_name].set_ylabel('Distance from GT y axis (m)')
        ax[algo_name].legend()

        ax[algo_name].set_title(algo_name)

        dist.plot_rotation(ax_rot[algo_name], ROTATION_SPAN)
        ax_rot[algo_name].set_ylabel('N of samples')
        ax_rot[algo_name].set_xlabel('Distance from GT (rad)')


    span = max(compute_span(ax['ICP']), compute_span(ax['NDT'])) * 1.2
    for algo_name in ['ICP', 'NDT']:
        error_f = TranslationErrorFunction(dataset, algo[algo_name], READING, REFERENCE, 500, n_samples_r=200)

        if args.cache.lower() in POSITIVE_STRINGS:
            error_f.load(cache_file[algo_name])
        else:
            error_f.compute(span, ROTATION_SPAN[1] - ROTATION_SPAN[0], file=cache_file[algo_name])
            error_f.read_data(cache_file[algo_name])

        error_f.plot(fig, ax[algo_name], colorbar=False)

        error_f_rot_ax = ax_rot[algo_name].twinx()
        error_f.plot_rotation(error_f_rot_ax)
        error_f_rot_ax.set_ylabel('Objective function')

    plt.savefig('dist-over-error.png')
    plt.show()
