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


def error_landscape_of_pair(pair, icp_algo):
    reading_fifo = random_fifo('.qpc')
    reference_fifo = random_fifo('.qpc')
    config_fifo = random_fifo('.yaml')

    cmd_string = ('recov_icp_error_landscape -reading {} -reference {} -ground_truth {}'
                  ' -config {} -nx {} -ny {} -nicp {}'
                  ' -icp_output {} -center -delta {}').format(
                      reading_fifo,
                      reference_fifo,
                      shlex.quote(json.dumps(pair.ground_truth().tolist())),
                      config_fifo,
                      10,
                      10,
                      5,
                      '/tmp/toto.json',
                      0.11
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
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.dataset, args.reading, args.reference)

    icp_algo = IcpAlgorithm()
    response = error_landscape_of_pair(pair, icp_algo)

    print(response)


if __name__ == '__main__':
    cli()
