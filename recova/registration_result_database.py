import argparse
import json
import multiprocessing
import numpy as np
import os
import pathlib
import re

from recov.datasets import create_registration_dataset
from recov.util import ln_se3

from recova.alignment import IdentityAlignmentAlgorithm
from recova.clustering import compute_distribution, CenteredClusteringAlgorithm
from recova.file_cache import FileCache
from recova.util import eprint
from recova.merge_json_result import merge_result_files
from recova.pointcloud import to_homogeneous
from recova.registration_dataset import lie_vectors_of_registrations


class RegistrationPair:
    def __init__(self, database_root, dataset, reading, reference):
        self.root = database_root
        self.dataset = dataset
        self.reading = reading
        self.reference = reference
        self.cache = FileCache(self.directory_of_pair / 'cache')

    def __str__(self):
        return 'Registration Pair: {}'.format(self.pair_id)

    @property
    def pair_id(self):
        return '{}-{:02d}-{:02d}'.format(self.dataset, self.reading, self.reference)


    @property
    def directory_of_pair(self):
        pair_folder = self.pair_id

        if not (self.root / pair_folder).exists() or not (self.root / pair_folder / 'raw').exists():
            os.makedirs(str(self.root / pair_folder / 'raw'))

        return self.root / pair_folder


    def accept_raw_file(self, filename):
        p = pathlib.Path(filename)
        dest = str(self.directory_of_pair / 'raw' / p.name)
        eprint('{} to {}'.format(p, dest))
        os.rename(str(p), dest)


    def import_pointclouds(self, pointcloud_dataset):
        """Import the reading and the reference that were used to generate the results from a pointcloud_dataset."""
        reading_file = self.directory_of_pair / 'reading.json'
        reference_file = self.directory_of_pair / 'reference.json'

        with reading_file.open('w') as f:
            json.dump(pointcloud_dataset.points_of_cloud(self.reading).tolist(), f)

        with reference_file.open('w') as f:
            json.dump(pointcloud_dataset.points_of_cloud(self.reference).tolist(), f)


    def pair_exists(self):
        return self.directory_of_pair.exists()


    def merge_raw_results(self):
        list_of_files = []

        for f in (self.directory_of_pair / 'raw').iterdir():
            if f.suffix == '.json':
                list_of_files.append(str(f))

        with (self.directory_of_pair / 'registrations.json').open('w') as registration_file:
            merge_result_files(list_of_files, registration_file)


    def lie_matrix_of_results(self):
        if not self.pair_exists():
            raise RuntimeError('No results available for demanded registration pair')

        reg_dict = self.registration_dict()

        return lie_vectors_of_registrations(reg_dict)

    @property
    def registration_file(self):
        return self.directory_of_pair / 'registrations.json'


    def registration_dict(self):
        if not self.registration_file.exists():
            self.merge_raw_results()

        with self.registration_file.open() as f:
            registration_dict = json.load(f)

        return registration_dict


    def combined_realigned(self, combiner, aligner):
        combined_realigned_name = 'cb_{}_{}'.format(combiner, aligner)
        transform_name = 'cbt_{}_{}'.format(combiner, aligner)

        if not self.cache[combined_realigned_name]:
            combined_realigned, t = self.compute_combined_realigned(combiner, aligner)
            self.cache[combined_realigned_name] = combined_realigned.tolist()
            self.cache[transform_name] = t.tolist()

            return combined_realigned, t
        else:
            return self.cache[combined_realigned_name], np.array(self.cache[transform_name])


    def compute_combined_realigned(self, combiner, aligner):
        reading = self.points_of_reading()
        reference = self.points_of_reference()
        initial_estimate = self.initial_estimate()

        combined = combiner.compute(reading, reference, self.initial_estimate())

        T = aligner.align(combined)
        np_combined = np.array(combined)

        homo_combined = to_homogeneous(np_combined)

        aligned = np.dot(T, homo_combined).T

        return aligned, T


    def descriptor(self, combiner, aligner, binner, descriptor_algorithm):
        descriptor_name = '{}_{}_{}_{}'.format(combiner, aligner, binner, descriptor_algorithm)

        descriptor = self.cache[descriptor_name]
        if not descriptor:
            descriptor = self.compute_descriptor(combiner, aligner, binner, descriptor_algorithm)
            self.cache[descriptor_name] = descriptor

        return descriptor


    def compute_descriptor(self, combiner, aligner, binner, descriptor_algorithm):
        reading = self.points_of_reading()
        reference = self.points_of_reference()
        initial_estimate = self.initial_estimate()

        combined_realigned, _ = self.combined_realigned(combiner, aligner)

        binned = binner.compute(combined_realigned.tolist())
        descriptor = descriptor_algorithm.compute(combined_realigned, binned)

        return descriptor


    def initial_estimate(self):
        if not self.registration_file.exists():
            self.merge_raw_results()

        with self.registration_file.open() as f:
            results = json.load(f)
            initial_estimate = results['metadata']['initial_estimate_mean']

        return np.array(initial_estimate)

    def ground_truth(self):
        with self.registration_file.open() as f:
            results = json.load(f)
            ground_truth = results['metadata']['ground_truth']

        return np.array(ground_truth)


    def points_of_reading(self):
        reading_file = self.directory_of_pair / 'reading.json'
        with reading_file.open() as f:
            return json.load(f)


    def points_of_reference(self):
        reading_file = self.directory_of_pair / 'reference.json'
        with reading_file.open() as f:
            return json.load(f)


    def clustering_of_results(self, clustering_algorithm):
        cached_clustering = self.cache[clustering_algorithm.__repr__()]

        if not cached_clustering:
            clustering = self.compute_clustering(clustering_algorithm)
            self.cache[clustering_algorithm.__repr__()] = clustering
            return clustering
        else:
            return cached_clustering


    def covariance(self, clustering_algorithm=CenteredClusteringAlgorithm()):
        clustering = self.clustering_of_results(clustering_algorithm)
        clustering = compute_distribution(self.registration_dict(), clustering)

        return np.array(clustering['covariance_of_central'])


    def compute_clustering(self, clustering_algorithm):
        ground_truth = self.registration_dict()['metadata']['ground_truth']

        lie = self.lie_matrix_of_results()

        clustering_row = clustering_algorithm.cluster(lie, seed=ln_se3(np.array(ground_truth)))
        distribution = compute_distribution(self.registration_dict(), clustering_row)

        return distribution




class RegistrationPairDatabase:
    def __init__(self, database_root, exclude=None):
        self.root = pathlib.Path(database_root)
        self.exclude = (re.compile(exclude) if exclude else None)

    def import_file(self, path_to_file):
        try:
            with open(path_to_file) as result_file:
                registration_results = json.load(result_file)

                dataset = registration_results['metadata']['dataset']
                reading = registration_results['metadata']['reading']
                reference = registration_results['metadata']['reference']

                r = RegistrationPair(self.root, dataset, reading, reference)
                r.accept_raw_file(path_to_file)

        except OSError as e:
            print(e)
            print('OSError for {}'.format(path_to_file))

    def get_registration_pair(self, dataset, reading, reference):
        return RegistrationPair(self.root, dataset, reading, reference)

    def registration_pairs(self):
        pairs = []
        for d in self.root.iterdir():
            if d.is_dir():
                components = d.stem.split('-')

                dataset = components[0]
                reading = int(components[1])
                reference = int(components[2])

                if not self.exclude or not self.exclude.match(dataset):
                    pairs.append(RegistrationPair(self.root, dataset, reading, reference))

        return pairs

def import_pointclouds_of_one_pair(registration_pair, dataset_type, pointcloud_root):
    print(registration_pair)
    dataset = create_registration_dataset(dataset_type, pointcloud_root / registration_pair.dataset)
    registration_pair.import_pointclouds(dataset)


def import_files_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', type=str, help='The files to import')
    parser.add_argument('--root', help='Location of the registration result database', type=str)
    parser.add_argument('--pointcloud_root', help='Location of the point clouds designated by the pairs', type=str)
    parser.add_argument('--pointcloud_dataset_type', help='The type of pointcloud dataset we import pointclouds from', type=str)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.root)

    for registration_file in args.files:
        print(registration_file)
        db.import_file(registration_file)

    pointcloud_root = pathlib.Path(args.pointcloud_root)

    with multiprocessing.Pool() as pool:
        pool.starmap(import_pointclouds_of_one_pair, [(x, args.pointcloud_dataset_type, pointcloud_root) for x in db.registration_pairs()])
