
import json
import os
import numpy as np
import pathlib

from recov.censi import compute_icp
from recov.pointcloud_io import pointcloud_to_pcd, pointcloud_to_xyz, read_xyz_stream
from recov.registration_algorithm import IcpAlgorithm
from recova.file_cache import FileCache
from recova.merge_json_result import merge_result_files
from recova.registration_dataset import positions_of_registration_data
from recova.util import eprint

class RegistrationPair:
    def __init__(self, database_root, dataset, reading, reference, database):
        self.root = database_root
        self.dataset = dataset
        self.reading = reading
        self.reference = reference
        self.cache = FileCache(self.directory_of_pair / 'cache')
        self.memory_cache = {}
        self.database = database

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
        def generate_transform():
            if use_odometry:
                algo = IcpAlgorithm()
                initial_estimate = pointcloud_dataset.odometry_estimate(self.reading, self.reference)
                transform, _ = compute_icp(self.database.reading_pcd(self.dataset, self.reading), self.database.reference_pcd(self.dataset, self.reference), initial_estimate, algo)
            else:
                transform = pointcloud_dataset.ground_truth(self.reading, self.reference)

            return transform

        eprint('Generating transform')
        self.cache.get_or_generate('transform', generate_transform)
        eprint('Transform generated for {}'.format(repr(self)))


    def transform(self):
        return self.cache['transform']


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
        return self.database.reading_pcd(self.dataset, self.reading)

    def path_to_reference_pcd(self):
        return self.database.reference_pcd(self.dataset, self.reference)

    def points_of_reading(self):
        return self.database.get_reading(self.dataset, self.reading)


    def points_of_reference(self):
        return self.database.get_reference(self.dataset, self.reference)


    def clustering_of_results(self, clustering_algorithm):
        cached_clustering = self.cache[clustering_algorithm.__repr__()]

        if not cached_clustering:
            clustering = self.compute_clustering(clustering_algorithm)
            self.cache[clustering_algorithm.__repr__()] = clustering
            return clustering
        else:
            return cached_clustering


    def compute_clustering(self, clustering_algorithm):
        ground_truth = self.registration_dict()['metadata']['ground_truth']

        lie = self.lie_matrix_of_results()

        clustering_row = clustering_algorithm.cluster(lie, seed=ln_se3(np.array(ground_truth)))
        distribution = compute_distribution(self.registration_dict(), clustering_row)

        return distribution

    def normals_of_reading(self):
        return self.database.normals_of_reading(self.dataset, self.reading)

    def normals_of_reference(self):
        return self.database.normals_of_reference(self.dataset, self.reference)

    def eigenvalues_of_reading(self):
        return self.database.eigenvalues_of_reading(self.dataset, self.reading)

    def eigenvalues_of_reference(self):
        return self.database.eigenvalues_of_reference(self.dataset, self.reference)
