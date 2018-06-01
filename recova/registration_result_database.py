import argparse
import json
import multiprocessing
import numpy as np
import os
import pathlib
import re
import subprocess
import sys

from io import StringIO

from recov.censi import compute_icp
from recov.datasets import create_registration_dataset
from recov.pointcloud_io import pointcloud_to_pcd, read_xyz_stream, pointcloud_to_xyz
from recov.registration_algorithm import IcpAlgorithm
from recov.util import ln_se3

from recova.alignment import IdentityAlignmentAlgorithm
from recova.clustering import compute_distribution, CenteredClusteringAlgorithm, IdentityClusteringAlgorithm,  clustering_algorithm_factory
from recova.covariance import covariance_algorithm_factory
from recova.file_cache import FileCache
from recova.util import eprint, run_subprocess
from recova.merge_json_result import merge_result_files
from recova.pointcloud import to_homogeneous
from recova.registration_dataset import lie_vectors_of_registrations, positions_of_registration_data


class RegistrationPair:
    def __init__(self, database_root, dataset, reading, reference):
        self.root = database_root
        self.dataset = dataset
        self.reading = reading
        self.reference = reference
        self.cache = FileCache(self.directory_of_pair / 'cache')

        self._points_of_reading = None
        self._points_of_reference = None

        self._normals = {}

    def __str__(self):
        return 'Registration Pair: {}'.format(self.pair_id)

    def __repr__(self):
        return self.pair_id

    @property
    def pair_id(self):
        return '{}-{:04d}-{:04d}'.format(self.dataset, self.reading, self.reference)


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


    def import_pointclouds(self, pointcloud_dataset, use_odometry=False):
        """Import the reading and the reference that were used to generate the results from a pointcloud_dataset."""
        reading_file = self.directory_of_pair / 'reading.xyz'
        reference_file = self.directory_of_pair / 'reference.xyz'

        with reading_file.open('w') as f:
            pointcloud_to_xyz(pointcloud_dataset.points_of_reading(self.reading), f)

        with reference_file.open('w') as f:
            pointcloud_to_xyz(pointcloud_dataset.points_of_reference(self.reference), f)


        if use_odometry:
            algo = IcpAlgorithm()
            initial_estimate = pointcloud_dataset.odometry_estimate(self.reading, self.reference)
            transform, _ = compute_icp(self.path_to_reading_pcd(), self.path_to_reference_pcd(), initial_estimate, algo)
        else:
            transform = pointcloud_dataset.ground_truth(self.reading, self.reference)

        np.save(self.directory_of_pair / 'transform.npy', transform)


    def transform(self):
        return np.load(self.directory_of_pair / 'transform.npy')


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

        return positions_of_registration_data(reg_dict)

    @property
    def registration_file(self):
        return self.directory_of_pair / 'registrations.json'


    def registration_dict(self):
        if not self.registration_file.exists():
            self.merge_raw_results()

        with self.registration_file.open() as f:
            registration_dict = json.load(f)

        return registration_dict


    def initial_estimate(self):
        if not self.registration_file.exists():
            self.merge_raw_results()

        with self.registration_file.open() as f:
            results = json.load(f)
            initial_estimate = results['metadata']['initial_estimate_mean']

        return np.array(initial_estimate)

    def ground_truth(self):
        return self.transform()

    def path_to_reading_pcd(self):
        path_to_pcd = self.directory_of_pair / 'reading.pcd'

        if not path_to_pcd.exists():
            pointcloud_to_pcd(self.points_of_reading(), str(path_to_pcd))

        return path_to_pcd


    def path_to_reference_pcd(self):
        path_to_pcd = self.directory_of_pair / 'reference.pcd'

        if not path_to_pcd.exists():
            pointcloud_to_pcd(self.points_of_reference(), str(path_to_pcd))

        return path_to_pcd


    def points_of_reading(self):
        if self._points_of_reading is None:
            reading_file = self.directory_of_pair / 'reading.xyz'
            with reading_file.open() as f:
                reading_points = read_xyz_stream(f)
                self._points_of_reading = reading_points

        return self._points_of_reading


    def points_of_reference(self):
        if self._points_of_reference is None:
            reference_file = self.directory_of_pair / 'reference.xyz'

            with reference_file.open() as f:
                reference_points = read_xyz_stream(f)
                self._points_of_reference = reference_points

        return self._points_of_reference


    def clustering_of_results(self, clustering_algorithm):
        cached_clustering = self.cache[clustering_algorithm.__repr__()]

        if not cached_clustering:
            clustering = self.compute_clustering(clustering_algorithm)
            self.cache[clustering_algorithm.__repr__()] = clustering
            return clustering
        else:
            return cached_clustering


    def covariance(self, clustering_algorithm=CenteredClusteringAlgorithm()):
        raise NotImplementedError('use of covariance() is deprecated. Use the CovarianceComputationAlgorithms instead.')


    def compute_clustering(self, clustering_algorithm):
        ground_truth = self.registration_dict()['metadata']['ground_truth']

        lie = self.lie_matrix_of_results()

        clustering_row = clustering_algorithm.cluster(lie, seed=ln_se3(np.array(ground_truth)))
        distribution = compute_distribution(self.registration_dict(), clustering_row)

        return distribution


    def _normals_of_pcd(self, pcd, k=18):
        cmd_template = 'normals_of_cloud -pointcloud {} -k {}'
        cmd_string = cmd_template.format(pcd, k)

        eprint(cmd_string)
        response = run_subprocess(cmd_string)
        stream = StringIO(response)
        normals = read_xyz_stream(stream)

        return normals


    def normals_of_cloud(self, label):
        """
        Computes the normal of the pointcloud designated by label.
        label can be reading_normals or reference_normals.
        """
        cache_file = self.directory_of_pair / (label + '.xyz')
        pcd_of_label = {
            'reading_normals': self.path_to_reading_pcd(),
            'reference_normals': self.path_to_reference_pcd()
        }

        if label in self._normals:
            # We have the normals in memory.
            return self._normals[label]
        elif cache_file.exists():
            # We have the normals in cache.
            with cache_file.open() as f:
                normals = read_xyz_stream(f)

            self._normals[label] = normals
            return normals
        else:
            # We need to compute the normals.
            normals = self._normals_of_pcd(pcd_of_label[label])

            # Cache them.
            with cache_file.open('w') as f:
                pointcloud_to_xyz(normals, f)
                f.flush()

            # Save them in memory.
            self._normals[label] = normals
            return normals


    def normals_of_reading(self):
        return self.normals_of_cloud('reading_normals')


    def normals_of_reference(self):
        return self.normals_of_cloud('reference_normals')




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

        return (dataset, reading, reference)


    def create_pair(self, location, reading, reference):
        pair = RegistrationPair(self.root, location, reading, reference)

        return pair


    def get_registration_pair(self, dataset, reading, reference):
        pair = RegistrationPair(self.root, dataset, reading, reference)

        if not pair.pair_exists():
            raise RuntimeError('Registration pair {} does not exist'.format(str(pair)))

        return pair

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

        pairs = sorted(pairs, key=lambda x: x.pair_id)

        return pairs

def import_pointclouds_of_one_pair(registration_pair, database, dataset_type, pointcloud_root):
    print(registration_pair)

    dataset = create_registration_dataset(dataset_type, pointcloud_root / registration_pair.dataset)
    registration_pair.import_pointclouds(dataset)


def import_one_husky_pair(db, location, reading, reference, dataset):
    eprint('{}: {} {}'.format(location, reading, reference))
    registration_pair = db.create_pair(location, reading, reference)
    registration_pair.import_pointclouds(dataset, use_odometry=True)


def import_husky_pointclouds_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database_root', type=str, help='The root of the pair database.')
    parser.add_argument('husky_dataset_root', type=str, help='Where to take the husky pointclouds')
    parser.add_argument('location', type=str, help='The label of the husky dataset')
    parser.add_argument('valid_scan_window_begin', type=float, help='For every map, take the scans that came valid_scan_window_begin seconds after the map was published.')
    parser.add_argument('valid_scan_window_end', type=float, help='For every map, take the scans that came valid_scan_window_end seconds before the map was published.')
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database_root)
    dataset = create_registration_dataset('husky', args.husky_dataset_root)

    pairs_to_fetch = []
    for reference in range(dataset.n_references()):
        pairs_to_fetch.extend(dataset.find_pairs_by_delay(reference, args.valid_scan_window_begin, args.valid_scan_window_end))

    with multiprocessing.Pool() as p:
        p.starmap(import_one_husky_pair, [(db, args.location, x[0], x[1], dataset) for x in pairs_to_fetch])



def import_files_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', type=str, help='The files to import')
    parser.add_argument('--root', help='Location of the registration result database', type=str)
    parser.add_argument('--pointcloud_root', help='Location of the point clouds designated by the pairs', type=str)
    parser.add_argument('--pointcloud_dataset_type', help='The type of pointcloud dataset we import pointclouds from', type=str, default='ethz')
    parser.add_argument('--pointcloud_only', help='Only do the pointcloud importation', action='store_true')
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.root)

    added_pairs_ids = set()

    if not args.pointcloud_only:
        for registration_file in args.files:
            print(registration_file)
            pair_id = db.import_file(registration_file)
            added_pairs_ids.add(pair_id)

    pointcloud_root = pathlib.Path(args.pointcloud_root)

    with multiprocessing.Pool() as pool:
        pool.starmap(import_pointclouds_of_one_pair, [(x, db, args.pointcloud_dataset_type, pointcloud_root) for x in db.registration_pairs()])


def distribution_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database_root', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('--covariance', type=str, help='The covariance estimation algorithm to use. <sampling|censi>', default='sampling')
    parser.add_argument('--clustering', type=str, help='The name of the clustering algorithm used by some sampling covariance algorithms.', default='identity')
    args = parser.parse_args()

    database = RegistrationPairDatabase(args.database_root)
    pair = database.get_registration_pair(args.dataset, args.reading, args.reference)

    clustering_algo = clustering_algorithm_factory(args.clustering)
    covariance_algo = covariance_algorithm_factory(args.covariance)
    covariance_algo.clustering_algorithm = clustering_algo

    covariance = covariance_algo.compute(pair)

    output_dict = {
        'mean': pair.ground_truth().tolist(),
        'covariance': covariance.tolist()
    }

    json.dump(output_dict, sys.stdout)


if __name__ == '__main__':
    import_files_cli()
