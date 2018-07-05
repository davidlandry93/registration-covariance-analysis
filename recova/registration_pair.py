
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
from recova.util import eprint, run_subprocess, rotation_around_z_matrix, transform_points

class RegistrationPair:
    def __init__(self, database_root, dataset, reading, reference, database, rotation_around_z = 0.0):
        self.root = database_root
        self.dataset = dataset
        self.reading = reading
        self.reference = reference
        self.cache = FileCache(self.directory_of_pair / 'cache')
        self.cache.prefix = self.prefix_from_rotation(rotation_around_z)
        self.database = database
        self._rotation_around_z = rotation_around_z

    def __str__(self):
        return 'Registration Pair: {}'.format(self.pair_id)

    def __repr__(self):
        return self.pair_id

    @property
    def rotation_around_z(self):
        return self._rotation_around_z

    @rotation_around_z.setter
    def set_rotation_around_z(self, new_rotation):
        self._rotation_around_z = new_rotation
        self.cache.prefix = self.prefix_from_rotation()

    def prefix_from_rotation(self, rotation):
        return '{:05f}'.format(rotation)

    @property
    def pair_id(self):
        return '{}-{:04d}-{:04d}'.format(
            self.dataset,
            self.reading,
            self.reference)

    @property
    def directory_of_pair(self):
        pair_folder = self.root / self.pair_id
        raw_data_folder = self.root / self.pair_id / 'raw'

        if not pair_folder.exists():
            raise RuntimeError('Folder not found for registration pair {}'.format(str(self)))
        elif not raw_data_folder.exists():
            os.mkdir(str(raw_data_folder))

        return pair_folder


    def accept_raw_file(self, filename):
        p = pathlib.Path(filename)
        eprint(self.directory_of_pair)
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
        R = rotation_around_z_matrix(self.rotation_around_z)
        return np.dot(rotation_around_z_matrix(self.rotation_around_z), np.dot(self.cache['transform'], np.linalg.inv(R)))


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


    def points_of_reading(self):
        points = self.database.get_reading(self.dataset, self.reading)

        R = rotation_around_z_matrix(self.rotation_around_z)

        points = transform_points(points, R)

        return points


    def points_of_reference(self):
        points = self.database.get_reference(self.dataset, self.reference)
        points = transform_points(points, rotation_around_z_matrix(self.rotation_around_z))
        return points


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

    def overlap(self):
        mask_reading, mask_reference = self.overlapping_region()

        return (np.sum(mask_reading) + np.sum(mask_reference)) / (len(mask_reading) + len(mask_reference))


    def overlapping_region(self, radius=0.1):
        key = 'overlapping_region_{}'.format(radius)

        def compute_overlapping_region(radius):
            reading = self.points_of_reading()
            reference = self.points_of_reference()
            t = self.transform()

            input_dict = {
                'reading': reading.tolist(),
                'reference': reference.tolist(),
                't': t.tolist()
            }

            cmd_string = 'overlapping_region -radius {} -mask'.format(radius)
            eprint(cmd_string)
            response = run_subprocess(cmd_string, json.dumps(input_dict))

            return json.loads(response)

        response = self.cache.get_or_generate(key, lambda: compute_overlapping_region(radius))
        reading_mask = np.array(response['reading'], dtype=bool)
        reference_mask = np.array(response['reference'], dtype=bool)

        return reading_mask, reference_mask
