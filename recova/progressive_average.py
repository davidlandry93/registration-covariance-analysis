import argparse
import json
import matplotlib.pyplot as plt
import lieroy.se3 as se3
import numpy as np


from recov.pointcloud_io import pointcloud_to_vtk
from recova.density import density_of_points
from recova.distribution_to_vtk_ellipsoid import distribution_to_vtk_ellipsoid
from recova.registration_dataset import lie_vectors_of_registrations

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Registration results')
    parser.add_argument('output', type=str, help='Output folder')
    args = parser.parse_args()

    with open(args.input) as f:
        data_dict = json.load(f)


    lie = lie_vectors_of_registrations(data_dict)
    density = np.array(density_of_points(lie))

    group = np.zeros((len(lie), 4, 4))
    for i, l in enumerate(lie):
        group[i] = se3.exp(l)


    print(lie)
    print(group)
    print(density)

    sorted_densities = sorted(density)
    sample_densities = np.linspace((np.min(density)), (np.max(density)), 50)
    print(sample_densities)

    print(np.mean(lie, axis=0))

    for i in range(0, 4000, 100):
        d = sorted_densities[i]
        results_subset = group[density > sorted_densities[i]]

        if len(results_subset) > 10:
            mean, covariance = se3.gaussian_from_sample(results_subset)


            pointcloud_to_vtk(lie[density > d][:,0:3], 'points_{}'.format(i), data = {'density': np.ascontiguousarray(density[density > d])})
            distribution_to_vtk_ellipsoid(se3.log(mean)[0:3], covariance[0:3,0:3], 'ellipsoid_{}'.format(i))
            print(covariance[0:3,0:3])
