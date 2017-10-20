#!/usr/bin/env python3

import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from datasets import create_registration_dataset
from registration_algorithm import AlgorithmFactory
from test_algos import angle_around_z_of_so3
from util import run_subprocess

TEMP_RESULT_FILE = 'tmp.json'

class TranslationErrorFunction:
    def __init__(self, dataset, algo, reading, reference, n_samples_t, n_samples_r=1):
        self.algo = algo
        self.reading = dataset.path_of_cloud(reading)
        self.reference = dataset.path_of_cloud(reference)
        self.n_samples_t = n_samples_t
        self.n_samples_r = n_samples_r
        self.ground_truth = dataset.ground_truth(reading, reference)
        self.center_point = dataset.ground_truth(reading, reference)
        self.filter_center = dataset.center_filter_size

    def compute(self, span_t, span_r, at=None, file=TEMP_RESULT_FILE):
        center_of_plot = None
        if at is None:
            center_of_plot = self.center_point
        else:
            center_of_plot = at

        cmd_string  = ('./error_function -algorithm {} -reading {} '
                       '-reference {} -ground_truth \"{}\" -output {} '
                       '-n_samples_t {} -n_samples_r {} -filter_center {} '
                       '-span_t {} -span_r {} -algo_config {}').format(
                           self.algo.name,
                           self.reading,
                           self.reference,
                           json.dumps(center_of_plot.tolist()),
                           file,
                           self.n_samples_t,
                           self.n_samples_r,
                           self.filter_center,
                           span_t,
                           span_r,
                           self.algo.config
                       )

        print(cmd_string)

        response = run_subprocess(cmd_string)

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

        ax.scatter([[0.0]], [[0.0]], color='black', marker='v', label='Ground truth')

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

        fig.canvas.mpl_connect('button_press_event', self.handler)

        ax.legend()
        plt.show()

    def redraw(self, p1, p2):
        mid_x = (p1[0] + p2[0]) / 2.0 + self.error_function.ground_truth[0,3]
        mid_y = (p1[1] + p2[1]) / 2.0 + self.error_function.ground_truth[1,3]

        self.error_function.center_point[0:2, 3] = np.array([mid_x, mid_y])

        span_x = max(p1[0], p2[0]) - min(p1[0], p2[0])

        self.error_function.compute(span_x, 0.1)

        self.plot()


def plot_error_rotation(data):
    permutation = np.argsort(data[:,0])
    plt.plot(data[permutation,0], data[permutation,1])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('kind', help='The kind of dataset user. <oru|ethz>.', type=str)
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

    dataset = create_registration_dataset(args.kind, args.dataset)
    algo = AlgorithmFactory.create(args.algo)
    error_f = TranslationErrorFunction(dataset,
                                       algo,
                                       args.reading,
                                       args.reference,
                                       args.n_samples_t,
                                       args.n_samples_r)

    interactive_error_f = InteractiveErrorFunctionPlot(error_f)
    interactive_error_f.run(initial_span=args.span_t)
