import argparse
import json
import multiprocessing
import pathlib
import sys

from recov.datasets import create_registration_dataset
from recova.io import read_xyz
from recova.util import eprint
from recova_core import grid_pointcloud_separator


def occupancy_descriptor(bin, total_n_points):
    return len(bin) / total_n_points


def generate_descriptor(pointcloud):
    bins = grid_pointcloud_separator(pointcloud, 20., 20., 20., 10, 10, 10)
    descriptor = [occupancy_descriptor(bin, len(pointcloud)) for bin in bins]

    return descriptor


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
