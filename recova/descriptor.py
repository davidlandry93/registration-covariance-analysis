import argparse
import json
import multiprocessing
import numpy as np
import pathlib
import shlex
import subprocess
import sys
import tempfile

from recov.registration_algorithm import IcpAlgorithm
from recov.datasets import create_registration_dataset
from recov.pointcloud_io import read_xyz_stream
from recova.util import eprint, run_subprocess



class DescriptorAlgorithm:
    """A descriptor algorithm takes a bin of points and outputs a description of the contents of the bin."""
    def __init__(self):
        pass

    def compute(self, pointcloud, bins):
        raise NotImplementedError('DescriptorAlgorithms must implement method compute')

    def __repr__(self):
        raise NotImplementedError('DescriptorAlgorithms must implement __repr__')


class OccupancyGridDescriptor(DescriptorAlgorithm):
    def compute(self, pointcloud, bins):
        descriptor = [len(x) / len(pointcloud) for x in bins]
        return descriptor

    def __repr__(self):
        return 'occupancy_grid'


class MomentGridDescriptor(DescriptorAlgorithm):
    def compute(self, pointcloud, bins):
        bin_descriptors = np.empty((len(bins), 12))
        for i, b in enumerate(bins):
            if len(b) == 0:
                bin_descriptors[i] = np.zeros(12)
            else:
                points = np.array(b)
                first_moment = points.mean(axis=0)

                centered_points = points - first_moment
                second_moment = np.dot(centered_points.T, centered_points) / len(centered_points)

                bin_descriptors[i, 0:3] = first_moment
                bin_descriptors[i, 3:12] = second_moment.flatten()

        return bin_descriptors.flatten().tolist()


    def __repr__(self):
        return 'moment-grid'


def occupancy_descriptor(bin, total_n_points):
    return len(bin) / total_n_points



def generate_descriptor_worker(dataset, i, output_dir):
    pointcloud = dataset.points_of_cloud(i)

    filename = 'descriptor_{}_{}.json'.format(dataset.name, i)
    eprint('Generating descriptor for {}'.format(filename))
    descriptor = generate_descriptor(pointcloud)

    with (output_dir / filename).open('w') as output_file:
        json.dump(descriptor, output_file)



def generate_descriptors(dataset, output_dir=pathlib.Path('.')):
    with multiprocessing.Pool() as pool:
        pool.starmap(generate_descriptor_worker, [(dataset, i, output_dir) for i in range(dataset.n_clouds())])



def generate_descriptors_cli():
    """
    Generate the descriptors for all the clouds in a dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='The dataset for which we compute the descriptors')
    parser.add_argument('--output', type=str, help='Location where to output the descriptors', default='.')
    args = parser.parse_args()

    dataset = create_registration_dataset('ethz', pathlib.Path(args.dataset))
    output_path = pathlib.Path(args.output)

    generate_descriptors(dataset, output_path)



def cli():
    """
    Outputs a full facet containing the descriptor and it's generation parameters.
    """
    pointcloud = read_xyz(sys.stdin)
    descriptor = generate_descriptor(pointcloud)

    output_dict = {
        'metadata': {
            'what': 'descriptor'
        }
    }

    print(descriptor)


if __name__ == '__main__':
    cli()
