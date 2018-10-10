import argparse
import json
import os
import shlex
import subprocess
import yaml

from recov.registration_algorithm import IcpAlgorithm
from recov.pointcloud_io import pointcloud_to_qpc_file
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint, random_fifo


def error_landscape_of_pair(pair, icp_algo, nx=100, ny=100, s=0.02, nicp=100, axis1=0, axis2=1):
    reading_fifo = random_fifo('.qpc')
    reference_fifo = random_fifo('.qpc')
    config_fifo = random_fifo('.yaml')

    cmd_string = ('recov_icp_error_landscape -reading {} -reference {} -ground_truth {}'
                  ' -config {} -nx {} -ny {} -nicp {}'
                  ' -icp_output {} -delta {} -axis1 {} -axis2 {} -center').format(
                      reading_fifo,
                      reference_fifo,
                      shlex.quote(json.dumps(pair.ground_truth().tolist())),
                      config_fifo,
                      nx,
                      ny,
                      nicp,
                      '/tmp/toto.json',
                      s,
                      axis1,
                      axis2
                  )

    eprint(cmd_string)

    proc = subprocess.Popen(
        cmd_string,
        shell=True,
        stdin=None,
        stdout=subprocess.PIPE,
        universal_newlines=True
    )


    pointcloud_to_qpc_file(pair.points_of_reading(), reading_fifo)
    pointcloud_to_qpc_file(pair.points_of_reference(), reference_fifo)

    with open(config_fifo, 'w') as f:
        yaml.dump(icp_algo.config_dict(), f)

    response = proc.stdout.read()

    os.unlink(reading_fifo)
    os.unlink(reference_fifo)
    os.unlink(config_fifo)

    return json.loads(response)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str)
    parser.add_argument('dataset', help='Path to the dataset', type=str)
    parser.add_argument('reading', help='Index of the pointcloud to use as reading', type=int)
    parser.add_argument('reference', help='Index of the pointcloud to use as reference', type=int)
    parser.add_argument('-x', '--n-samples-x', help='Number of columns of sampling', type=int, default=100)
    parser.add_argument('-y', '--n-samples-y', help='Number of rows of sampling', type=int, default=100)
    parser.add_argument('-s', '--span', help='Size of one size of the square in m', type=float, default=0.05)
    parser.add_argument('-n', '--n-icp', help='N of icp samples to compute', type=int, default=30)
    parser.add_argument('-a1', '--axis1', default=0, type=int)
    parser.add_argument('-a2', '--axis2', default=1, type=int)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.dataset, args.reading, args.reference)

    icp_algo = IcpAlgorithm()
    icp_algo.initial_estimate_covariance = 0.01
    icp_algo.initial_estimate_covariance_rot = 0.01
    icp_algo.max_iteration_count = 120
    response = error_landscape_of_pair(
        pair,
        icp_algo,
        args.n_samples_x,
        args.n_samples_y,
        args.span,
        args.n_icp,
        args.axis1,
        args.axis2
    )

    print(json.dumps(response))


if __name__ == '__main__':
    cli()
