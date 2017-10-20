#!/usr/bin/env python3

import argparse
import json
import numpy as np

from datasets import create_registration_dataset
from registration_algorithm import AlgorithmFactory
from util import run_subprocess

def compute_ground_truth(dataset, reading, reference):
    return dataset.ground_truth(reading, reference)

def compute_estimate(dataset, reading, reference):
    return dataset.odometry_estimate(reading, reference)

def compute_transform(dataset, reading, reference, algorithm):
    algo = AlgorithmFactory.create(algorithm)
    algo.n_samples = 5
    distributions = algo.compute_covariance_with_dataset(dataset, reading, reference)

    return distributions['mean']

def visualize_transform(dataset, reading, reference, transform):
    cmd_string = './visualize_transform -reading {} -reference {} -transform \"{}\"'.format(
        dataset.path_of_cloud(reading),
        dataset.path_of_cloud(reference),
        json.dumps(transform.tolist())
    )

    run_subprocess(cmd_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('kind', help='The kind of dataset used. <oru|ethz>', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('transform', help='Which transform to visualize <icp|ndt|estimate|ground_truth|idendity>', type=str)
    args = parser.parse_args()

    dataset = create_registration_dataset(args.kind, args.dataset)

    transform = np.identity(4)
    if args.transform == 'ndt' or args.transform == 'icp':
        transform = compute_transform(dataset, args.reading, args.reference, args.transform)
    elif args.transform == 'estimate' or args.transform == 'est':
        transform = compute_estimate(dataset, args.reading, args.reference)
    elif args.transform == 'ground_truth' or args.transform == 'gt':
        transform = compute_ground_truth(dataset, args.reading, args.reference)
    elif args.transform == 'id' or args.transform == 'identity':
        pass
    else:
        raise ValueError('Unrecognized transform {}'.format(args.transform))

    print(transform)
    visualize_transform(dataset, args.reading, args.reference, transform)


