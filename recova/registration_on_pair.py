#!/usr/bin/env python3

import argparse
import numpy as np

from recov.censi import registration_and_censi_estimate_from_points
from recov.registration_algorithm import IcpAlgorithm

from recova.registration_result_database import RegistrationPairDatabase

from lieroy import se3

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str)
    parser.add_argument('location', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('--no-vtk-inspector', action='store_false')
    parser.add_argument('--dist-type', type=str, default='normal')
    parser.add_argument('--max-iter-count', type=int, default=60)
    parser.add_argument('--init-var-translation', type=float, default=0.05)
    parser.add_argument('--init-var-rotation', type=float, default=0.05)
    args = parser.parse_args()

    algo = IcpAlgorithm()
    algo.use_vtk_inspector(args.no_vtk_inspector)
    algo.max_iteration_count = args.max_iter_count
    algo.initial_estiamte_covariance = args.init_var_translation
    algo.initial_estimate_covariance_rot = args.init_var_rotation

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.location, args.reading, args.reference)

    covariance = np.identity(6)
    covariance[0:3,0:3] *= args.init_var_translation
    covariance[3:6,3:6] *= args.init_var_rotation
    initial_estimate = se3.sample_normal(pair.ground_truth(), covariance, n_samples=1)[0]

    print(initial_estimate)

    reg_result, censi_estimate = registration_and_censi_estimate_from_points(pair.points_of_reading(), pair.points_of_reference(), initial_estimate, algo)

    delta = np.linalg.inv(pair.ground_truth()) @ reg_result
    delta_lie = se3.log(delta)

    print('Error translation: {} Error rotation: {}'.format(np.linalg.norm(delta_lie[0:3]), np.linalg.norm(delta_lie[3:6])))


if __name__ == '__main__':
    cli()
