import argparse

import recov.pointcloud_io

from recova.registration_result_database import RegistrationPairDatabase

def pair_to_vtk_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str)
    parser.add_argument('location', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('--no-apply-transform', action='store_true')
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.location, args.reading, args.reference)

    if args.no_apply_transform:
        t = np.identity(4)
    else:
        t = pair.transform()

    recov.pointcloud_io.pointcloud_to_vtk(pair.points_of_reading(), 'reading', T=t)
    recov.pointcloud_io.pointcloud_to_vtk(pair.points_of_reference(), 'reference')

