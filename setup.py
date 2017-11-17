import io
import os
import sys

from setuptools import find_packages, setup

setup(
    name='recova',
    description='Data analysis tools for the registration covariance computations.',
    author='David Landry',
    author_email='davidlandry93@gmail.com',
    url='https://github.com/davidlandry93/registration-covariance-analysis',
    packages=find_packages(exclude='tests'),
    package_data={
        'recova': ['core.so']
    },
    install_requires=['numpy', 'pyevtk'],
    scripts = ['bin/all_registration_to_vtk_ellipsoid',
               'bin/meta_of_json'],
    entry_points = {
        'console_scripts': [
            'center_registrations = recova.registration_dataset:center_around_gt_cli',
            'clustering = recova.clustering:cli',
            'clustering_batch = recova.clustering_batch:cli',
            'clusterings2distributions = recova.clustering:compute_distributions_cli',
            'clusterings2vtk = recova.clustering:batch_to_vtk_cli',
            'distribution2vtk = recova.distribution_to_vtk_ellipsoid:cli',
            'find_central_cluster = recova.find_center_cluster:cli',
            'generate_cube = recova.pointcloud_gen.cube:cli',
            'json_cat = recova.json_util:json_cat_cli',
            'merge_json_result = recova.merge_json_result:cli',
            'recov_plot = recova.recov_plot:cli',
            'registration_heatmap = recova.single_experiment_heatmap:cli',
            'registration_2d_plot = recova.single_experiment_plot:cli',
            'registration2covariance = recova.covariance_of_registrations:cli',
            'registration2vtk = recova.results_to_vtk:cli',
            'registration2lie = recova.registration_dataset:registration2lie_cli',
            'trails2vtk = recova.trails_to_vtk:cli'
        ]
    },
)
