#!/usr/bin/env python3

import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import subprocess


from recov.datasets import create_registration_dataset
from recov.pointcloud_io import pointcloud_to_qpc_file
from recov.registration_algorithm import AlgorithmFactory

from recova.clustering import DensityThresholdClusteringAlgorithm
from recova.util import run_subprocess, random_fifo

from recova.registration_result_database import RegistrationPairDatabase

TEMP_RESULT_FILE = 'tmp.json'

class TranslationErrorFunction:
    def __init__(self, pair, algo, n_samples_t, n_samples_r=1):
        self.pair = pair
        self.algo = algo
        self.n_samples_t = n_samples_t
        self.n_samples_r = n_samples_r
        self.ground_truth = pair.transform()
        self.center_point = pair.transform()

    def clustered_results(self):
        clustering_algo = DensityThresholdClusteringAlgorithm(1e5, 100)
        results = self.pair.lie_matrix_of_results()
        clustering = clustering_algo.cluster(results)
        clustering = np.array(clustering['clustering'][0])
        clustered_results = results[clustering]

        return clustered_results


    def center_around_results(self, clustering_algo):
        clustered_results = self.clustered_results()

        min_x, max_x = clustered_results[:,0].min(), clustered_results[:,0].max()
        min_y, max_y = clustered_results[:,1].min(), clustered_results[:,1].max()

        print('X from {} to {}'.format(min_x, max_x))
        print('Y from {} to {}'.format(min_y, max_y))

        self.center_point[0,3] = (max_x + min_x) / 2.0
        self.center_point[1,3] = (max_y + min_y) / 2.0

        print(self.center_point)



    def compute(self, span_t, span_r, at=None, file=TEMP_RESULT_FILE):
        center_of_plot = None
        if at is None:
            center_of_plot = self.center_point
        else:
            center_of_plot = at

        reading_fifo = random_fifo()
        reference_fifo = random_fifo()

        cmd_string  = ('recov_error_function -algorithm {} -reading {} '
                       '-reference {} -ground_truth \"{}\" -output {} '
                       '-n_samples_t {} -n_samples_r {} '
                       '-span_t {} -span_r {} -algo_config {}').format(
                           self.algo.name,
                           reading_fifo,
                           reference_fifo,
                           json.dumps(center_of_plot.tolist()),
                           file,
                           self.n_samples_t,
                           self.n_samples_r,
                           span_t,
                           span_r,
                           self.algo.config
                       )

        print(cmd_string)

        proc = subprocess.Popen(
            cmd_string,
            shell=True,
            stdin=None,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )

        reading = self.pair.points_of_reading()
        reference = self.pair.points_of_reference()

        pointcloud_to_qpc_file(reading, reading_fifo)
        pointcloud_to_qpc_file(reference, reference_fifo)

        response = proc.stdout.read()



    def read_data(self, file=TEMP_RESULT_FILE):
        with open(file) as json_file:
            data = json.loads(json_file.read())

        self.translation_data = np.array(data['data']['translation'])
        self.rotation_data = np.array(data['data']['rotation'])
        self.rotation_data = self.rotation_data[np.argsort(self.rotation_data[:,0])]

        return data

    def plot(self, fig, ax, colorbar=True):
        xs = self.translation_data[:,0]
        ys = self.translation_data[:,1]
        zs = self.translation_data[:,2]

        xi = np.linspace(xs.min(), xs.max(), 1000)
        yi = np.linspace(ys.min(), ys.max(), 1000)
        zi = griddata((xs, ys), zs, (xi[None,:], yi[:,None]))

        xi = xi - self.ground_truth[0,3]
        yi = yi - self.ground_truth[1,3]

        contour = ax.contourf(xi,yi,zi, 40, cmap=plt.get_cmap('plasma_r'))
        if colorbar:
            fig.colorbar(contour)

        # ax.scatter([[0.0]], [[0.0]], color='black', marker='v', label='Ground truth')
        results = self.clustered_results()
        ax.scatter(results[:,0] - self.ground_truth[0,3], results[:,1] - self.ground_truth[1,3], s=0.5, color='black')

        return fig

    def plot_rotation(self, ax):
        gt = angle_around_z_of_so3(self.ground_truth[0:3,0:3])

        ax.plot(self.rotation_data[:,0] - gt, self.rotation_data[:,1], color='black')


class InteractiveErrorFunctionPlot:
    def __init__(self, error_function):
        self.error_function = error_function
        self.first_click = None

    def handler(self, e):
        print('x of data: {}, y of data: {}'.format(e.xdata, e.ydata))
        if self.first_click is None:
            self.first_click = (e.xdata, e.ydata)
        else:
            first_click = self.first_click
            self.first_click = None
            self.redraw(first_click, (e.xdata, e.ydata))

    def run(self, initial_span=2.0):
        self.error_function.compute(initial_span, 0.1)

        self.plot()

    def plot(self):
        self.error_function.read_data()

        plt.close('all')
        fig, ax = plt.subplots()
        self.error_function.plot(fig, ax)

        ax.set_xlabel('Distance from ground truth x axis (m)')
        ax.set_ylabel('Distance from ground truth y axis (m)')
        ax.set_aspect('equal')

        fig.canvas.mpl_connect('button_press_event', self.handler)

        ax.legend()
        plt.show()

    def redraw(self, p1, p2):
        mid_x = (p1[0] + p2[0]) / 2.0 + self.error_function.ground_truth[0,3]
        mid_y = (p1[1] + p2[1]) / 2.0 + self.error_function.ground_truth[1,3]

        print('Mid point: ({},{})'.format(mid_x, mid_y))

        self.error_function.center_point[0:2, 3] = np.array([mid_x, mid_y])

        span_x = max(p1[0], p2[0]) - min(p1[0], p2[0])

        self.error_function.compute(span_x, 0.1)

        self.plot()


def plot_error_rotation(data):
    permutation = np.argsort(data[:,0])
    plt.plot(data[permutation,0], data[permutation,1])
    plt.show()

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('algo', help='The algorithm to assess', type=str, default="icp")
    parser.add_argument('--cache', help='Use the results in cache', type=str, default="false")
    parser.add_argument('--span_t', help='The span to plot in translation', type=float, default=1.0)
    parser.add_argument('--span_r', help='The span of the data to plot in rotation', type=float, default=0.1)
    parser.add_argument('--n_samples_t', help='The number of samples to collect for translation data', type=int, default=500)
    parser.add_argument('--n_samples_r', help='The number of samples to collect for rotation data', type=int, default=100)

    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.dataset, args.reading, args.reference)

    algo = AlgorithmFactory.create(args.algo)
    error_f = TranslationErrorFunction(pair,
                                       algo,
                                       args.n_samples_t,
                                       args.n_samples_r)

    clustering = DensityThresholdClusteringAlgorithm(1e5, 100)
    error_f.center_around_results(clustering)

    interactive_error_f = InteractiveErrorFunctionPlot(error_f)
    interactive_error_f.run(initial_span=args.span_t)



if __name__ == '__main__':
    cli()
