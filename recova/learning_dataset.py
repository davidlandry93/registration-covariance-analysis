
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
from recova.descriptor.mask import OverlapMaskGenerator
from recova.descriptor.descriptor import ConcatDescriptorAlgo, Descriptor, MomentsDescriptorAlgo, NormalHistogramDescriptionAlgo
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


def generate_one_example(registration_pair, descriptor, covariance_algo):
    eprint(registration_pair)

    descriptor_start = time.time()
    descriptor = descriptor.compute(registration_pair)
    eprint('Descriptor took {} seconds'.format(time.time() - descriptor_start))
    covariance = covariance_algo.compute(registration_pair)
    eprint('Example took {} seconds'.format(time.time() - descriptor_start))

    return (descriptor, covariance)


def generate_examples_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Where to store the examples', default='dataset.json')
    parser.add_argument('--input', type=str, help='Where the registration results are stored', default='.', required=True)
    parser.add_argument('--exclude', type=str, help='Regex of names of datasets to exclude', default='gazebo_winter|wood_summer')
    parser.add_argument('-j', '--n_cores', type=int, help='N of cores to use for the computation', default=8)
    args = parser.parse_args()

    np.set_printoptions(linewidth=120)

    db = RegistrationPairDatabase(args.input, args.exclude)
    output_path = pathlib.Path(args.output)


    # clustering_algorithm = CenteredClusteringAlgorithm(0.2)
    # clustering_algorithm.rescale = True

    clustering_algorithm = IdentityClusteringAlgorithm()
    covariance_algo = SamplingCovarianceComputationAlgorithm(clustering_algorithm)

    registration_pairs = db.registration_pairs()



    mask_generator = OverlapMaskGenerator()
    description_algo = ConcatDescriptorAlgo([MomentsDescriptorAlgo(), NormalHistogramDescriptionAlgo()])
    descriptor = Descriptor(mask_generator, description_algo)

    with multiprocessing.Pool(args.n_cores) as pool:
        examples = pool.starmap(generate_one_example, [(x, descriptor, covariance_algo) for x in registration_pairs])

    xs = []
    ys = []
    for p in examples:
        x, y = p
        xs.append(x.tolist())
        ys.append(y.tolist())


    with open(args.output, 'w') as dataset_file:
        json.dump({
            'metadata': {
                'what': 'learning_dataset',
                'date': str(datetime.datetime.today()),
                'descriptor': str(descriptor),
                'clustering': str(clustering_algorithm),
                'descriptor_labels': descriptor.labels()
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



def compute_one_summary_line(registration_pair, covariance_algo):
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


def dataset_summary_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the managed registration result database')
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.input)

    # clustering_algorithm = CenteredClusteringAlgorithm(0.05, k=100)
    # clustering_algorithm.seed_selector = 'localized'
    # clustering_algorithm.rescale = True

    clustering_algorithm = IdentityClusteringAlgorithm()

    covariance_algorithm = SamplingCovarianceComputationAlgorithm(clustering_algorithm)

    # covariance_algorithm = CensiCovarianceComputationAlgorithm()

    with multiprocessing.Pool() as p:
        rows = p.starmap(compute_one_summary_line, [(x, covariance_algorithm) for x in db.registration_pairs()])

    writer = csv.DictWriter(sys.stdout, ['dataset', 'reading', 'reference', 'cluster_distance', 'outlier_ratio', 'condition_number', 'trace'])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
