
import argparse
import copy
import csv
import datetime
import json
import multiprocessing
import numpy as np
import pathlib
import re
import sys
import time
import tqdm

from lieroy import se3

from recov.datasets import create_registration_dataset
from recova.alignment import IdentityAlignmentAlgorithm, PCAlignmentAlgorithm
from recova.clustering import CenteredClusteringAlgorithm, IdentityClusteringAlgorithm, DensityThresholdClusteringAlgorithm, OutlierFilterClusteringAlgorithm
from recova.covariance import SamplingCovarianceComputationAlgorithm, CensiCovarianceComputationAlgorithm
from recova.descriptor.factory import descriptor_factory
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint, nearestPD, parallel_starmap_progressbar


def vectorize_covariance(cov_matrix):
    pd_matrix = nearestPD(cov_matrix)

    error = np.linalg.norm(cov_matrix - pd_matrix)

    lower_triangular = np.linalg.cholesky(pd_matrix)

    vector_of_cov = []
    for i in range(6):
        for j in range(i + 1):
            vector_of_cov.append(lower_triangular[i, j])


    return vector_of_cov


def generate_one_example(registration_pair, descriptor, covariance_algo,descriptor_only=False, rotation=0.0):
    registration_pair.rotation_around_z = rotation

    eprint(registration_pair)
    descriptor_start = time.time()
    descriptor = descriptor.compute(registration_pair)
    eprint('Descriptor took {} seconds'.format(time.time() - descriptor_start))

    if not descriptor_only:
        covariance = covariance_algo.compute(registration_pair)
    else:
        covariance = None

    eprint('Example took {} seconds'.format(time.time() - descriptor_start))

    return (descriptor, np.array(covariance))


def generate_examples_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Where to store the examples', default='dataset.json')
    parser.add_argument('--input', type=str, help='Where the registration results are stored', default='.', required=True)
    parser.add_argument('--exclude', type=str, help='Regex of names of datasets to exclude', default='')
    parser.add_argument('-j', '--n_cores', type=int, help='N of cores to use for the computation', default=8)
    parser.add_argument('-c', '--config', type=str, help='Path to a json config for the descriptor.')
    parser.add_argument('--descriptor-only', action='store_true', help='Generate only the descriptor.')
    parser.add_argument('--rotations', '-r', nargs='+', type=float, default=[0.0])
    args = parser.parse_args()

    np.set_printoptions(linewidth=120)

    db = RegistrationPairDatabase(args.input, args.exclude)
    registration_pairs = db.registration_pairs()

    output_path = pathlib.Path(args.output)

    # clustering = CenteredClusteringAlgorithm(0.005, k=20)
    # clustering.seed_selector = 'localized'
    # clustering.rescale = True

    # clustering = DensityThresholdClusteringAlgorithm(threshold=1e3, k=100)
    clustering = OutlierFilterClusteringAlgorithm()
    covariance_algo = SamplingCovarianceComputationAlgorithm(clustering_algorithm=clustering)

    with open(args.config) as f:
        descriptor_config = json.load(f)

    descriptor = descriptor_factory(descriptor_config)

    eprint('Using descriptor: {}'.format(repr(descriptor)))
    eprint('Generating with rotations: {}'.format(args.rotations))

    examples = []
    pairs = []
    for x in registration_pairs:
        examples.extend([(x, descriptor, covariance_algo, args.descriptor_only, r) for r in args.rotations])
        pairs.extend([{'dataset': x.dataset, 'reading': x.reading, 'reference': x.reference, 'rotation': r} for r in args.rotations])


    results = parallel_starmap_progressbar(generate_one_example, examples, n_cores=args.n_cores)

    # results = [generate_one_example(*x) for x in examples]


    xs = []
    ys = []
    for p in results:
        x, y = p
        xs.append(x.tolist())
        if not args.descriptor_only:
            ys.append(y.tolist())

    output_dict = {
            'metadata': {
                'what': 'learning_dataset',
                'date': str(datetime.datetime.today()),
                'descriptor': str(descriptor),
                'covariance_algo': str(covariance_algo),
                'descriptor_labels': descriptor.labels(),
                'descriptor_config': descriptor_config,
                'filter': args.exclude
            },
            'statistics': {
                'n_examples': len(xs)
            },
            'data': {
                'pairs': pairs,
                'xs': xs,
            }
        }

    if not args.descriptor_only:
        output_dict['data']['ys'] = ys

    with open(args.output, 'w') as dataset_file:
        json.dump(output_dict, dataset_file)


def compute_one_summary_line(registration_pair, covariance_algo):
    eprint(registration_pair)
    covariance = covariance_algo.compute(registration_pair)

    d = {
        'dataset': registration_pair.dataset,
        'reading': registration_pair.reading,
        'reference': registration_pair.reference,
        'condition_number': np.linalg.cond(covariance),
        'trace': np.trace(covariance),
    }

    eprint('%s done' % str(registration_pair))

    return d


def filter_dataset(learning_dataset, filter_regex, keep=False):
    """
    Arguments
    learning_dataset: A full json learning dataset (with a data.pairs, data.xs and data.ys)
    filter_regex: A compiled regex that matches the dataset names we want to filter out
    keep: If true the regex matches what we want to keep. Otherwise it matches what we want to filter out.
    """
    mask = np.zeros(len(learning_dataset['data']['pairs']), dtype=bool)

    for i in range(len(learning_dataset['data']['pairs'])):
        ma = filter_regex.match(learning_dataset['data']['pairs'][i]['dataset'])
        mask[i] = (ma is not None) == keep

    return mask




def dataset_summary_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the managed registration result database')
    parser.add_argument('--density-filter', type=float, default=0.0)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.input)

    clustering_algorithm = CenteredClusteringAlgorithm(0.05, k=100)
    clustering_algorithm.seed_selector = 'localized'
    clustering_algorithm.rescale = True

    # clustering_algorithm = DensityThresholdClusteringAlgorithm(args.density_filter, k=100)
    covariance_algorithm = SamplingCovarianceComputationAlgorithm(clustering_algorithm)

    # covariance_algorithm = CensiCovarianceComputationAlgorithm()

    with multiprocessing.Pool(1) as p:
        rows = p.starmap(compute_one_summary_line, [(x, covariance_algorithm) for x in db.registration_pairs()])
        writer = csv.DictWriter(sys.stdout, ['dataset', 'reading', 'reference', 'cluster_distance', 'outlier_ratio', 'condition_number', 'trace'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    generate_examples_cli()
