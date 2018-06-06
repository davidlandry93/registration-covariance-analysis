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

from recov.datasets import create_registration_dataset
from recov.pointcloud_io import pointcloud_to_pcd, read_xyz_stream, pointcloud_to_xyz
from recov.util import ln_se3

from recova.alignment import IdentityAlignmentAlgorithm
from recova.clustering import compute_distribution, CenteredClusteringAlgorithm, IdentityClusteringAlgorithm,  clustering_algorithm_factory
from recova.covariance import covariance_algorithm_factory
from recova.file_cache import FileCache
from recova.util import eprint, run_subprocess, parallel_starmap_progressbar
from recova.merge_json_result import merge_result_files
from recova.pointcloud import to_homogeneous
from recova.registration_dataset import lie_vectors_of_registrations, positions_of_registration_data
from recova.registration_pair import RegistrationPair



class RegistrationPairDatabase:
    def __init__(self, database_root, exclude=None):
        self.root = pathlib.Path(database_root)

        cache_dir = self.root / 'cache'
        pointcloud_dir = self.root / 'pointclouds'
        if self.root.exists():
            cache_dir.mkdir(exist_ok=True)
            pointcloud_dir.mkdir(exist_ok=True)
        else:
            raise RuntimeError('Path {} does not exist'.format(self.root))

        self.exclude = (re.compile(exclude) if exclude else None)
        self.cache = FileCache(cache_dir)

    def pointcloud_dir(self):
        return self.root / 'pointclouds'

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
        pair = RegistrationPair(self.root, location, reading, reference, self)

        return pair


    def get_registration_pair(self, dataset, reading, reference):
        pair = RegistrationPair(self.root, dataset, reading, reference, self)

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

    def import_reading(self, location, index, dataset):
        points = dataset.points_of_reading(index)
        self.import_pointcloud(points, '{}_{}_{}'.format(location, 'reading', index))

    def import_reference(self, location, index, dataset):
        points = dataset.points_of_reference(index)
        self.import_pointcloud(points, '{}_{}_{}'.format(location, 'reference', index))

    def import_pointcloud(self, points, label):
        _ = self.cache.get_or_generate(label, lambda: points)

    def get_reading(self, location, index):
        label = '{}_{}_{}'.format(location, 'reading', index)
        return self.cache[label]

    def get_reference(self, location, index):
        label = '{}_{}_{}'.format(location, 'reference', index)
        return self.cache[label]

    def reference_pcd(self, location, index):
        path_to_pcd = self.pointcloud_dir() / '{}_{}_{}.pcd'.format(location, 'reference', index)

        if not path_to_pcd.exists():
            pointcloud_to_pcd(self.get_reference(location, index), str(path_to_pcd))

        return path_to_pcd

    def reading_pcd(self, location, index):
        path_to_pcd = self.pointcloud_dir() / '{}_{}_{}.pcd'.format(location, 'reading', index)

        if not path_to_pcd.exists():
            pointcloud_to_pcd(self.get_reading(location, index), str(path_to_pcd))

        return path_to_pcd

    def normals_of_reading(self, location, index):
        label = '{}_{}_{}_{}'.format(location, 'reading', index, 'normals')
        self.cache.get_or_generate(label, lambda: self.normals_of_pcd(self.reading_pcd()))

    def normals_of_reference(self, location, index):
        label = '{}_{}_{}_{}'.format(location, 'reference', index, 'normals')
        self.cache.get_or_generate(label, lambda: self.normals_of_pcd(self.reference_pcd()))

    def normals_of_pcd(self, pcd, k=18):
        cmd_template = 'normals_of_cloud -pointcloud {} -k {}'
        cmd_string = cmd_template.format(pcd, k)

        eprint(cmd_string)
        response = run_subprocess(cmd_string)
        stream = StringIO(response)
        normals = read_xyz_stream(stream)

        return normals



def import_pointclouds_of_one_pair(registration_pair, database, dataset_type, pointcloud_root):
    print(registration_pair)

    db.import_reading(registration_pair.dataset, registration_pair.reading, dataset)
    db.import_reference(registration_pair.dataset, registration_pair.reference, dataset)

    dataset = create_registration_dataset(dataset_type, pointcloud_root / registration_pair.dataset)
    registration_pair.import_pointclouds(dataset)



def import_one_husky_pair(db, location, reading, reference, dataset):
    db.import_reading(location, reading, dataset)
    db.import_reference(location, reference, dataset)

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
    parser.add_argument('-j', '--n-cores', help='Number of parallel processes to spawn', type=int)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database_root)
    dataset = create_registration_dataset('husky', args.husky_dataset_root)


    pairs_to_fetch = []
    for reference in range(dataset.n_references()):
        pairs_to_fetch.extend(dataset.find_pairs_by_delay(reference, args.valid_scan_window_begin, args.valid_scan_window_end))

    all_tuples = [(db, args.location, x[0], x[1], dataset) for x in pairs_to_fetch]
    # with multiprocessing.Pool() as p:
    #     p.starmap(import_one_husky_pair, [(db, args.location, x[0], x[1], dataset) for x in pairs_to_fetch])

    parallel_starmap_progressbar(import_one_husky_pair, all_tuples, n_cores=args.n_cores)

    # for tup in all_tuples:
    #     import_one_husky_pair(*tup)


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
    import_husky_pointclouds_cli()
