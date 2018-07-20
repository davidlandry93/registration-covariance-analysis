import argparse
import concurrent.futures
import functools
import json
import multiprocessing
import numpy as np
import os
import pathlib
import random
import re
import subprocess
import sys
import tqdm

from io import StringIO

import lieroy

from recov.datasets import create_registration_dataset
from recov.pointcloud_io import pointcloud_to_pcd, read_xyz_stream, pointcloud_to_xyz
from recov.util import ln_se3

from recova.alignment import IdentityAlignmentAlgorithm
from recova.clustering import compute_distribution, CenteredClusteringAlgorithm, IdentityClusteringAlgorithm,  clustering_algorithm_factory
from recova.covariance import covariance_algorithm_factory
from recova.file_cache import FileCache
from recova.util import eprint, run_subprocess, parallel_starmap_progressbar, rotation_around_z_matrix, transform_points
from recova.merge_json_result import merge_result_files
from recova.pointcloud import to_homogeneous
from recova.registration_dataset import lie_vectors_of_registrations
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
        self.pointcloud_cache = FileCache(pointcloud_dir)

    def pointcloud_dir(self):
        return self.root / 'pointclouds'

    def import_file(self, path_to_file):
        try:
            with open(path_to_file) as result_file:
                registration_results = json.load(result_file)

                dataset = registration_results['metadata']['dataset']
                reading = registration_results['metadata']['reading']
                reference = registration_results['metadata']['reference']

                if not self.pair_exists(dataset, reading, reference):
                    pair = self.create_pair(dataset, reading, reference)
                else:
                    pair = self.get_registration_pair(dataset, reading, reference)

                pair.accept_raw_file(path_to_file)

        except OSError as e:
            print(e)
            print('OSError for {}'.format(path_to_file))

        return (dataset, reading, reference)

    def pair_exists(self, location, reading, reference):
        pair_path = self.root / '{}-{:04d}-{:04d}'.format(location, reading, reference)
        return pair_path.exists()


    def create_pair(self, location, reading, reference):
        os.makedirs(str(self.root / '{}-{:04d}-{:04d}'.format(location, reading, reference) / 'raw'))
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
            if d.is_dir() and d.stem != 'pointclouds' and d.stem != 'cache':
                components = d.stem.split('-')

                dataset = components[0]
                reading  = int(components[1])
                reference = int(components[2])

                if not self.exclude or not self.exclude.match(dataset):
                    pairs.append(RegistrationPair(self.root, dataset, reading, reference, self))

        pairs = sorted(pairs, key=lambda x: x.pair_id)

        return pairs

    def import_reading(self, location, index, dataset):
        label = '{}_{}_{}'.format(location, 'reading', index)
        _ = self.pointcloud_cache.get_or_generate(label, lambda: dataset.points_of_reading(index))

    def import_reference(self, location, index, dataset):
        label = '{}_{}_{}'.format(location, 'reference', index)
        _ = self.pointcloud_cache.get_or_generate(label, lambda: dataset.points_of_reference(index))


    def get_reading(self, location, index):
        label = '{}_{}_{}'.format(location, 'reading', index)
        return self.pointcloud_cache[label]

    def get_reference(self, location, index):
        label = '{}_{}_{}'.format(location, 'reference', index)
        return self.pointcloud_cache[label]


    def reference_pcd(self, location, index):
        pointcloud_label = '{}_{}_{}'.format(location, 'reference', index)
        return self.pointcloud_pcd(pointcloud_label, self.get_reference, rotation_around_z)


    def reading_pcd(self, location, index):
        pointcloud_label = '{}_{}_{}'.format(location, 'reading', index)
        return self.pointcloud_pcd(pointcloud_label, self.get_reading, rotation_around_z)

    def pointcloud_pcd(self, pointcloud_label, pointcloud_lambda):
        label = pointcloud_label
        path_to_pcd = self.pointcloud_dir() / label + '.pcd'

        if not path_to_pcd.exists():
            points = pointcloud_lambda()
            pointcloud_to_pcd(transformed, path_to_pcd)

        return path_to_pcd


    def normals_of_reading(self, location, index):
        label = '{}_{}_{}_{}'.format(location, 'reading', index, 'normals')
        normals = self.cache.get_or_generate(label, lambda: self.normals_of_pcd(self.reading_pcd(location, index)))

        return normals

    def normals_of_reference(self, location, index):
        label = '{}_{}_{}_{}'.format(location, 'reference', index, 'normals')
        normals = self.cache.get_or_generate(label, lambda: self.normals_of_pcd(self.reference_pcd(location, index)))
        return normals

    def normals_of_pcd(self, pcd, k=18):
        cmd_template = 'normals_of_cloud -pointcloud {} -k {}'
        cmd_string = cmd_template.format(pcd, k)


        eprint(cmd_string)
        response = run_subprocess(cmd_string)
        stream = StringIO(response)
        normals = read_xyz_stream(stream)

        return normals

    def eigenvalues_of_reading(self, location, index):
        label = '{}_{}_{}_{}'.format(location, 'reading', index, 'eigvals')
        return self.cache.get_or_generate(label, lambda: self.eigenvalues_of_pcd(self.reading_pcd(location, index)))

    def eigenvalues_of_reference(self, location, index):
        label = '{}_{}_{}_{}'.format(location, 'reference', index, 'eigvals')
        return self.cache.get_or_generate(label, lambda: self.eigenvalues_of_pcd(self.reference_pcd(location, index)))

    def eigenvalues_of_pcd(self, pcd):
        cmd_string = 'eigenvalues_of_cloud -cloud {}'.format(pcd)
        result = run_subprocess(cmd_string)
        result_dict = json.loads(result)

        cloud_eig_vals = np.array(result_dict)

        return cloud_eig_vals.T


def import_one_husky_pair(db, location, reading, reference, dataset):
    eprint('{}: {} {}'.format(location, reading, reference))
    registration_pair = db.create_pair(location, reading, reference)
    registration_pair.import_pointclouds(dataset, use_odometry=True)


def starmap_wrapper(f, tup):
    return f(*tup)

def import_reading(location, x, dataset, db):
    db.import_reading(location, x, dataset)
    _ = db.reading_pcd(location, x)

def import_reference(location, x, dataset, db):
    db.import_reference(location, x, dataset)
    _ = db.reference_pcd(location, x)

def compute_data_reading(location, x, db):
    _ = db.normals_of_reading(location, x)
    _ = db.eigenvalues_of_reading(location, x)

def compute_data_reference(location, x, db):
    _ = db.normals_of_reference(location, x)
    _ = db.eigenvalues_of_reference(location, x)

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

    all_readings = [(args.location, x, dataset, db) for x in range(dataset.n_readings())]
    all_references = [(args.location, x, dataset, db) for x in range(dataset.n_references())]

    pairs_to_fetch = []
    for reference in range(dataset.n_references()):
        pairs_to_fetch.extend(dataset.find_pairs_by_delay(reference, args.valid_scan_window_begin, args.valid_scan_window_end))


    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_cores) as executor:
        futures = []
        for cloud in all_readings:
            futures.append(executor.submit(import_reading, *cloud))
        for cloud in all_references:
            futures.append(executor.submit(import_reference, *cloud))

        concurrent.futures.wait(futures)

        for cloud in all_readings:
            futures.append(executor.submit(compute_data_reading, args.location, cloud[1], db))

        for cloud in all_references:
            futures.append(executor.submit(compute_data_reference, args.location, cloud[1], db))

        concurrent.futures.wait(futures)

        for pair in pairs_to_fetch:
            futures.append(executor.submit(import_one_husky_pair, db, args.location, pair[0], pair[1], dataset))

        concurrent.futures.wait(futures)

def import_pointclouds_of_one_pair(registration_pair, dataset):
    registration_pair.import_pointclouds(dataset)
    _ = registration_pair.registration_dict()


def import_files_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='*', type=str, help='The files to import')
    parser.add_argument('--root', help='Location of the registration result database', type=str)
    parser.add_argument('--pointcloud_root', help='Location of the point clouds designated by the pairs', type=str)
    parser.add_argument('--pointcloud_dataset_type', help='The type of pointcloud dataset we import pointclouds from', type=str, default='ethz')
    parser.add_argument('--pointcloud_only', help='Only do the pointcloud importation', action='store_true')
    parser.add_argument('-j', '--n-cores', default=8, type=int)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.root)

    added_pairs_ids = set()

    if not args.pointcloud_only:
        for registration_file in args.files:
            print(registration_file)
            pair_id = db.import_file(registration_file)
            added_pairs_ids.add(pair_id)

    pointcloud_root = pathlib.Path(args.pointcloud_root)

    readings = {}
    references = {}
    for pair in db.registration_pairs():
        if pair.dataset not in readings:
            readings[pair.dataset] = set([pair.reading])
        else:
            readings[pair.dataset].add(pair.reading)

        if pair.dataset not in references:
            references[pair.dataset] = set([pair.reference])
        else:
            references[pair.dataset].add(pair.reference)


    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_cores) as executor:
        fs = []
        progress_bar = tqdm.tqdm(total=5*len(db.registration_pairs()), file=sys.stdout)
        for dataset_name in readings:
            dataset = create_registration_dataset(args.pointcloud_dataset_type, pointcloud_root / dataset_name)

            for reading in readings[dataset_name]:
                future = executor.submit(import_reading, dataset_name, reading, dataset, db)
                future.add_done_callback(lambda _: progress_bar.update())
                fs.append(future)

            for reference in references[dataset_name]:
                future = executor.submit(import_reference, dataset_name, reference, dataset, db)
                future.add_done_callback(lambda _: progress_bar.update())
                fs.append(future)

        concurrent.futures.wait(fs)

        fs = []
        for dataset_name in readings:
            for reading in readings[dataset_name]:
                eprint('{}: {}'.format(dataset_name, reading))
                future = executor.submit(compute_data_reading, dataset_name, reading, db)
                future.add_done_callback(lambda _: progress_bar.update())
                fs.append(future)

            for reference in references[dataset_name]:
                eprint('{}: {}'.format(dataset_name, reference))
                future = executor.submit(compute_data_reference, dataset_name, reference, db)
                future.add_done_callback(lambda _: progress_bar.update())
                fs.append(future)

        concurrent.futures.wait(fs)

        fs = []
        for pair in db.registration_pairs():
            pointcloud_dataset = create_registration_dataset(args.pointcloud_dataset_type, pointcloud_root / pair.dataset)

            future = executor.submit(import_pointclouds_of_one_pair, pair, pointcloud_dataset)
            future.add_done_callback(lambda _: progress_bar.update())
            fs.append(future)

        concurrent.futures.wait(fs)



def distribution_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database_root', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('--covariance', type=str, help='The covariance estimation algorithm to use. <sampling|censi>', default='sampling')
    parser.add_argument('--clustering', type=str, help='The name of the clustering algorithm used by some sampling covariance algorithms.', default='identity')
    parser.add_argument('-r', '--rotation', type=float, default=0.0, help='Rotation around the z axis to apply to the dataset')
    args = parser.parse_args()

    database = RegistrationPairDatabase(args.database_root)
    pair = database.get_registration_pair(args.dataset, args.reading, args.reference)
    pair.rotation_around_z = args.rotation

    clustering_algo = clustering_algorithm_factory(args.clustering)
    covariance_algo = covariance_algorithm_factory(args.covariance)
    covariance_algo.clustering_algorithm = clustering_algo

    covariance = covariance_algo.compute(pair)

    output_dict = {
        'mean': pair.transform().tolist(),
        'covariance': covariance.tolist()
    }

    json.dump(output_dict, sys.stdout)


if __name__ == '__main__':
    import_husky_pointclouds_cli()
