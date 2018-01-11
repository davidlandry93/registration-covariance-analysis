import argparse
import json
import multiprocessing
import pathlib
import subprocess
import sys

from recov.registration_algorithm import IcpAlgorithm
from recov.datasets import create_registration_dataset
from recova.io import read_xyz
from recova.util import eprint
from recova_core import grid_pointcloud_separator



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




class BinningAlgorithm:
    """A binning alogrithm takes a pointcloud and puts it into bins we can compute a descriptor."""
    def __init__(self):
        pass

    def compute(self, reading, reference):
        raise NotImplementedError('Binning algorithms must implement method compute')

    def __repr__(self):
        raise NotImplementedError('BinningAlgorithms must implement __repr__')



class GridBinningAlgorithm(BinningAlgorithm):
    def __init__(self, spanx, spany, spanz, nx, ny, nz):
        self.spanx = spanx
        self.spany = spany
        self.spanz = spanz
        self.nx = nx
        self.ny = ny
        self.nz = nz


    def compute(self, pointcloud):
        command_string = 'grid_pointcloud_separator -spanx {} -spany {} -spanz {} -nx {} -ny {} -nz {}'.format(
            self.spanx,
            self.spany,
            self.spanz,
            self.nx,
            self.ny,
            self.nz
        )
        eprint(command_string)

        response = subprocess.check_output(
            command_string,
            universal_newlines=True,
            shell=True,
            input=json.dumps(pointcloud)
        )

        return json.loads(response)

    def __repr__(self):
        return 'grid-{:.4f}-{:.4f}-{:.4f}-{}-{}-{}'.format(self.spanx, self.spany, self.spanz, self.nx, self.ny, self.nz)



class PointcloudCombiner:
    """A pointcloud combiner takes the reading and outputs a single pointcloud which we use for learning."""
    def compute(self, reading, reference, t):
        raise NotImplementedError('PointcloudCombiners must implement compute')

    def __repr__(self):
        raise NotImplementedError('PoincloudCombiners must implement __repr__')



class ReferenceOnlyCombiner(PointcloudCombiner):
    def compute(self, reading, reference, t):
        return reference

    def __repr__(self):
        return 'ref-only'


class OverlappingRegionCombiner(PointcloudCombiner):
    """A pointcloud combiner that registers the point clouds and returns the overlapping region."""
    def compute(self, reading, reference, initial_estimate):
        icp = IcpAlgorithm()
        t = icp.register(reading, reference, initial_estimate)

    def __repr__(self):
        return 'ref-only_icp'



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
