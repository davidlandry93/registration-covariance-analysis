
import argparse
import copy
import csv
import datetime
import json
import multiprocessing
import numpy as np
import pathlib
import time
import sys

from lie import se3

from recov.datasets import create_registration_dataset
from recova.alignment import IdentityAlignmentAlgorithm, PCAlignmentAlgorithm
from recova.clustering import CenteredClusteringAlgorithm, IdentityClusteringAlgorithm
from recova.covariance import SamplingCovarianceComputationAlgorithm, CensiCovarianceComputationAlgorithm
from recova.descriptor import OccupancyGridDescriptor, MomentGridDescriptor
from recova.binning import GridBinningAlgorithm
from recova.combiner import ReferenceOnlyCombiner, OverlappingRegionCombiner
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint, nearestPD


def vectorize_covariance(cov_matrix):
    pd_matrix = nearestPD(cov_matrix)

    error = np.linalg.norm(cov_matrix - pd_matrix)

    lower_triangular = np.linalg.cholesky(pd_matrix)

    vector_of_cov = []
    for i in range(6):
        for j in range(i + 1):
            vector_of_cov.append(lower_triangular[i, j])


    return vector_of_cov


def generate_one_example(registration_pair, combining_algorithm, alignment_algorithm, binning_algorithm, descriptor_algorithm, clustering_algorithm):
    eprint(registration_pair)
    descriptor = registration_pair.descriptor(combining_algorithm, alignment_algorithm, binning_algorithm, descriptor_algorithm)
    covariance = registration_pair.covariance(clustering_algorithm)

    _, t = registration_pair.combined_realigned(combining_algorithm, alignment_algorithm)

    adj_of_t = se3.adjoint(t)
    rotated_covariance = np.dot(adj_of_t, np.dot(covariance, adj_of_t.T))
    vectorized_covariance = vectorize_covariance(rotated_covariance)

    return (descriptor, vectorized_covariance)


def generate_examples_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Where to store the examples', default='dataset.json')
    parser.add_argument('--input', type=str, help='Where the registration results are stored', default='.', required=True)
    args = parser.parse_args()

    np.set_printoptions(linewidth=120)

    db = RegistrationPairDatabase(args.input)
    output_path = pathlib.Path(args.output)

    combiner = OverlappingRegionCombiner()
    aligner = IdentityAlignmentAlgorithm()
    binning_algorithm = GridBinningAlgorithm(10., 10., 5., 3, 3, 3)
    descriptor_algorithm = MomentGridDescriptor()

    clustering_algorithm = CenteredClusteringAlgorithm(0.2)
    clustering_algorithm.rescale = True

    registration_pairs = db.registration_pairs()

    with multiprocessing.Pool() as pool:
        examples = pool.starmap(generate_one_example, [(x, combiner, aligner, binning_algorithm, descriptor_algorithm, clustering_algorithm) for x in registration_pairs])

    xs, ys = zip(*examples)

    with open(args.output, 'w') as dataset_file:
        json.dump({
            'metadata': {
                'what': 'learning_dataset',
                'date': str(datetime.datetime.today()),
                'combiner': str(combiner),
                'binner': str(binning_algorithm),
                'descriptor': str(descriptor_algorithm),
                'clustering': str(clustering_algorithm)
            },
            'statistics': {
                'n_examples': len(xs)
            },
            'data': {
                'pairs': [{
                    'dataset': x.dataset,
                    'reading': x.reading,
                    'reference': x.reference} for x in registration_pairs],
                'xs': xs,
                'ys': ys,
            }
        }, dataset_file)


def cello_examples_of_registration_pair(pair, combiner, aligner, binner, descriptor, clustering):
    descriptor = pair.descriptor(combiner,
                                 aligner,
                                 binner,
                                 descriptor)

    print(pair)
    clustering = pair.clustering_of_results(clustering)
    clustering = np.array(clustering['clustering'][0])

    errors = pair.lie_matrix_of_results()
    clustered_errors = errors[clustering]

    print(clustered_errors.shape)

    return descriptor, clustered_errors


def generate_cello_dataset_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Where to store the examples', default='dataset.json')
    parser.add_argument('--input', type=str, help='Where the registration results are stored', default='.', required=True)
    parser.add_argument('--exclude', type=str, help='Regex of names of datasets to exclude', default='gazebo_winter|wood_summer')
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.input, args.exclude)
    output_path = pathlib.Path(args.output)


    combiner = OverlappingRegionCombiner()
    aligner = IdentityAlignmentAlgorithm()
    binning_algorithm = GridBinningAlgorithm(10., 10., 5., 3, 3, 3)
    descriptor_algorithm = MomentGridDescriptor()

    # clustering_algorithm = CenteredClusteringAlgorithm(0.2)
    # clustering_algorithm.rescale = True

    clustering_algorithm = IdentityClusteringAlgorithm()

    registration_pairs = db.registration_pairs()

    with multiprocessing.Pool() as pool:
        examples = pool.starmap(cello_examples_of_registration_pair, [(x, combiner, aligner, binning_algorithm, descriptor_algorithm, clustering_algorithm) for x in registration_pairs])

    predictors = []
    all_errors = []

    for predictor, errors in examples:
        predictors.append(predictor)
        all_errors.append(errors.tolist())

    output_document = {
        'metadata': {
            'what': 'cello_dataset',
            'date': str(datetime.datetime.today()),
            'combiner': str(combiner),
            'binner': str(binning_algorithm),
            'descriptor': str(descriptor_algorithm),
            'clustering': str(clustering_algorithm)
        },
        'statistics': {
            'n_examples': len(predictors)
        },
        'data': {
            'predictors': predictors,
            'errors': all_errors
        }
    }

    with open(args.output, 'w') as json_file:
        json_file.write(json.dumps(output_document))


def compute_one_summary_line(registration_pair, covariance_algo):
    covariance = covariance_algo.compute(registration_pair)

    eprint(np.linalg.norm(covariance - covariance.T))

    d = {
        'dataset': registration_pair.dataset,
        'reading': registration_pair.reading,
        'reference': registration_pair.reference,
        'condition_number': np.linalg.cond(covariance),
        'trace': np.trace(covariance),
    }

    eprint('%s done' % str(registration_pair))

    return d


def dataset_summary_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the managed registration result database')
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.input)

    clustering_algorithm = CenteredClusteringAlgorithm(0.05, k=100)
    clustering_algorithm.seed_selector = 'localized'
    clustering_algorithm.rescale = True
    clustering_algorithm = IdentityClusteringAlgorithm()

    covariance_algorithm = SamplingCovarianceComputationAlgorithm(clustering_algorithm)

    # covariance_algorithm = CensiCovarianceComputationAlgorithm()

    with multiprocessing.Pool() as p:
        rows = p.starmap(compute_one_summary_line, [(x, covariance_algorithm) for x in db.registration_pairs()])

    writer = csv.DictWriter(sys.stdout, ['dataset', 'reading', 'reference', 'cluster_distance', 'outlier_ratio', 'condition_number', 'trace'])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
