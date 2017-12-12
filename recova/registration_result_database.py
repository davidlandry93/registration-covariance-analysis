import argparse
import json
import numpy as np
import os
import pathlib

from recov.util import ln_se3

from recova.clustering import compute_distribution
from recova.util import eprint
from recova.merge_json_result import merge_result_files
from recova.registration_dataset import lie_vectors_of_registrations


class RegistrationResult:
    def __init__(self, database_root, dataset, reading, reference):
        self.root = database_root
        self.dataset = dataset
        self.reading = reading
        self.reference = reference

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


    def registration_dict(self):
        registration_file = self.directory_of_pair / 'registrations.json'
        if not registration_file.exists():
            self.merge_raw_results()

        with registration_file.open() as f:
            registration_dict = json.load(f)

        return registration_dict


    def covariance(self, clustering_algorithm, radius=0.2):
        clustering_directory = self.directory_of_pair / 'clustering'

        if not clustering_directory.exists():
            clustering_directory.mkdir()

        clustering_file = clustering_directory / (str(clustering_algorithm) + '.json')
        if not clustering_file.exists():
            clustering = self.compute_clustering(clustering_algorithm, radius=radius)
            clustering = compute_distribution(self.registration_dict(), clustering)

            with clustering_file.open('w') as jsonfile:
                json.dump(clustering, jsonfile)
        else:
            try:
                with clustering_file.open() as jsonfile:
                    clustering = json.load(jsonfile)
                    eprint('Using cached clustering for {}'.format(self))
            except ValueError:
                print('Error decoding clustering {} for {}'.format(clustering_algorithm, self))


        return np.array(clustering['covariance_of_central'])


    def compute_clustering(self, clustering_algorithm, radius=0.2):
        ground_truth = self.registration_dict()['metadata']['ground_truth']

        lie = self.lie_matrix_of_results()

        clustering_row = clustering_algorithm.cluster(lie, seed=ln_se3(np.array(ground_truth)))
        distribution = compute_distribution(self.registration_dict(), clustering_row)

        return distribution




class RegistrationResultDatabase:
    def __init__(self, database_root):
        self.root = pathlib.Path(database_root)

    def import_file(self, path_to_file):
        try:
            with open(path_to_file) as result_file:
                registration_results = json.load(result_file)

                dataset = registration_results['metadata']['dataset']
                reading = registration_results['metadata']['reading']
                reference = registration_results['metadata']['reference']

                r = RegistrationResult(self.root, dataset, reading, reference)
                r.accept_raw_file(path_to_file)

        except OSError as e:
            print(e)
            print('OSError for {}'.format(path_to_file))

    def get_registration_pair(self, dataset, reading, reference):
        return RegistrationResult(self.root, dataset, reading, reference)

    def registration_pairs(self):
        pairs = []
        for d in self.root.iterdir():
            if d.is_dir():
                components = d.stem.split('-')

                dataset = components[0]
                reading = int(components[1])
                reference = int(components[2])
                pairs.append(RegistrationResult(self.root, dataset, reading, reference))

        return pairs


def import_files_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', type=str, help='The files to import')
    parser.add_argument('--root', help='Location of the registration result database', type=str)
    args = parser.parse_args()

    db = RegistrationResultDatabase(args.root)

    for registration_file in args.files:
        db.import_file(registration_file)
