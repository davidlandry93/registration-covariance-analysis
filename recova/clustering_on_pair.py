import argparse
import numpy as np
from lieroy import se3

from recov.pointcloud_io import pointcloud_to_vtk
from recova.clustering import clustering_algorithm_factory
from recova.registration_result_database import RegistrationPairDatabase

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str)
    parser.add_argument('location', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('--algo', '-a', type=str, default='centered')
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-r', type=float, default=0.05)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.location, args.reading, args.reference)

    clustering_algo = clustering_algorithm_factory(args.algo)
    clustering_algo.k = args.k
    clustering_algo.radius = args.r
    clustering_algo.seed_selector = 'localized'

    lie_results = pair.lie_matrix_of_results()
    response = clustering_algo.cluster(lie_results, seed=se3.log(pair.ground_truth()))

    cluster = np.array(response['clustering'][0])

    pointcloud_to_vtk(lie_results[:,0:3], 'results')
    pointcloud_to_vtk(lie_results[cluster][:,0:3], 'clustered')

if __name__ == '__main__':
    cli()
