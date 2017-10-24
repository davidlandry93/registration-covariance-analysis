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
    install_requires=['numpy', 'pyevtk'],
    scripts = ['bin/all_registration_to_vtk_ellipsoid'],
    entry_points = {
        'console_scripts': [
            'clustering = recova.clustering_dbscan:cli',
            'distribution2vtk = recova.distribution_to_vtk_ellipsoid:cli',
            'generate_cube = recova.pointcloud_gen.cube:cli',
            'merge_json_result = recova.merge_json_result:cli',
            'registration_heatmap = recova.single_experiment_heatmap:cli',
            'registration_2d_plot = recova.single_experiment_plot:cli',
            'registration2vtk = recova.results_to_vtk:cli',
            'trails2vtk = recova.trails_to_vtk:cli'
        ]
    }
)
