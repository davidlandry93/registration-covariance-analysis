import argparse
import sys

from recova.io import read_xyz
from recova_core import grid_pointcloud_separator

def occupancy_descriptor(bin, total_n_points):
    return len(bin) / total_n_points

def cli():
    pointcloud = read_xyz(sys.stdin)

    print(pointcloud)

    bins = grid_pointcloud_separator(pointcloud, 20., 20., 20., 10, 10, 10)

    descriptor = [occupancy_descriptor(bin, len(pointcloud)) for bin in bins]

    print(descriptor)


if __name__ == '__main__':
    cli()
