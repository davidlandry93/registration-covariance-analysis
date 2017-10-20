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
            'merge_json_result = recova.merge_json_result:cli',
            'trails2vtk = recova.trails_to_vtk:cli'
        ]
    }
)
