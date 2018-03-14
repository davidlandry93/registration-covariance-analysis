import argparse
import json
import numpy as np
import scipy as sp

from recova.pointcloud import to_homogeneous

from sklearn.decomposition import PCA

class AlignmentAlgorithm():
    def __init__(self):
        self._transform = None

    def align(pointcloud):
        """Takes a combined pointcloud and a covariance and realigs them in a principled manner"""
        raise NotImplemented("Alignment Algorithms must implement align")

    def __repr__(self):
        raise NotImplemented("Alignment algorithms must implement __repr__")

    @property
    def transform(self):
        if self._transform is None:
            raise RuntimeError('AlignmentAlgorithm should have defined a transform by now')

        return self._transform


class IdentityAlignmentAlgorithm(AlignmentAlgorithm):
    def init(self):
        self._transform = np.identity(4)

    def align(self, pointcloud):
        self._transform = np.identity(4)
        return np.identity(4)

    def __repr__(self):
        return 'identity'


class PCAlignmentAlgorithm(AlignmentAlgorithm):
    def align(self, pointcloud):
        np_pointcloud = np.array(pointcloud)

        pca = PCA(n_components=3)
        pca.fit(pointcloud)

        self._transform = np.identity(4)
        self._transform[0:3,0:3] = pca.components_
        self._transform[0:3,3] = np.mean(pointcloud, axis=1)[0:3]

        print('T: \n{}'.format(self._transform))

        return self._transform


    def __repr__(self):
        return 'principal-components'


def pca_alignment_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    with open(args.input) as jsonfile:
        pointcloud = json.load(jsonfile)

    pointcloud = np.array(pointcloud)

    aligner = PCAlignmentAlgorithm()
    T = aligner.align(pointcloud)

    homo_pointcloud = to_homogeneous(pointcloud)
    transformed = np.dot(T, homo_pointcloud)

    with open('realigned.json', 'w') as jsonfile:
        jsonfile.write(json.dumps(transformed.T.tolist()))

    with open('reference.json', 'w') as jsonfile:
        jsonfile.write(json.dumps(pointcloud.tolist()))
