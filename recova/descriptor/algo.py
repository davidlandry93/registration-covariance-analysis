import json
import numpy as np
import os
import shlex
import subprocess
import tempfile
import uuid

from recov.censi import censi_estimate_from_points
from recov.registration_algorithm import IcpAlgorithm
from recov.pointcloud_io import pointcloud_to_pcd, points_to_temporary_pcd, pointcloud_to_qpc_file

from recova.util import eprint, run_subprocess

class DescriptorAlgo:
    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError('Descriptor algorithms must implement __repr__ method')

    def compute(self, pair, reading_mask, reference_mask):
        raise NotImplementedError('Descriptor algorithms must implement compute method')

    def labels(self):
        raise NotImplementedError('Descriptor algorithms must implement labels method')


class ConcatDescriptorAlgo(DescriptorAlgo):
    def __init__(self, algos):
        self.algos = algos

    def __repr__(self):
        algo_reprs = [repr(x) for x in self.algos]

        return '_'.join(algo_reprs)

    def compute(self, pair, reading_mask, reference_mask):
        descriptors = []

        for algo in self.algos:
            descriptor = algo.compute(pair, reading_mask, reference_mask)
            descriptors.append(descriptor)

        return np.concatenate(descriptors)

    def labels(self):
        labels = []
        for algo in self.algos:
            labels.extend(algo.labels())
        return labels


class OccupancyDescriptorAlgo(DescriptorAlgo):
    def __init__(self):
        pass

    def __repr__(self):
        return 'occupancy'

    def compute(self, pair, reading_mask, reference_mask):
        return [(np.sum(reading_mask) + np.sum(reference_mask)) / float(len(reading_mask) + len(reference_mask))]

    def labels(self):
        return ['occupancy']



class MomentsDescriptorAlgo(DescriptorAlgo):
    def __init__(self):
        pass

    def __repr__(self):
        return 'moments'

    def compute(self, pair, reading_mask, reference_mask):
        reading = pair.points_of_reading()
        reference = pair.points_of_reference()

        reading = reading[reading_mask]
        reference = reference[reference_mask]

        points = np.vstack((reading, reference))

        if len(points) == 0:
            return np.zeros(12)

        descriptor = np.empty(12)
        mean = points.mean(axis=0)
        descriptor[0:3] = mean

        # Center the points before computing the second moment.
        points = points - descriptor[0:3]

        second_moment = np.dot(points.T, points) / len(points)
        descriptor[3:12] = second_moment.flatten()

        return descriptor


    def labels(self):
        labels = ['moment1_x', 'moment1_y', 'moment1_z']

        for dim1 in ['x', 'y', 'z']:
            for dim2 in ['x', 'y', 'z']:
                labels.append('moment2_{}{}'.format(dim1,dim2))

        return labels



class NormalsHistogramDescriptionAlgo(DescriptorAlgo):
    def __init__(self):
        pass

    def __repr__(self):
        return 'norm_histogram'

    def compute(self, pair, reading_mask, reference_mask):
        normals_reading = pair.normals_of_reading()
        normals_reference = pair.normals_of_reference()[reference_mask]

        normals_reading = pair.normals_of_reading()
        normals_reading = normals_reading[reading_mask]


        normals_reference = pair.normals_of_reference()
        normals_reference = normals_reference[reference_mask]

        normals = np.vstack((normals_reading, normals_reference))

        ref_lines = self.reference_lines()
        histogram = np.zeros(len(ref_lines))

        distances = np.empty((len(normals), len(ref_lines)))
        for i, line in enumerate(ref_lines):
            distances[:,i] = self.point_to_line_distances(np.array([0.,0.,0.]), line, normals)


        mins = np.argmin(distances, axis=1)
        histogram = np.zeros(len(ref_lines))

        for minimum in mins:
            histogram[minimum] += 1.

        if np.sum(histogram) == 0.:
            return histogram
        else:
            return histogram / np.sum(histogram)


    def labels(self):
        return ['norm_line{}'.format(x) for x in range(len(self.reference_lines()))]

    def reference_lines(self):
        # Here we define the lines with two points.
        # For every line x1 is the origin.
        lines = []

        # Four lines parallel to the xy plane.
        lines.append([1., 0., 0.])
        lines.append([0., 1., 0.])
        lines.append([1., 1., 0.])
        lines.append([-1., 1., 0.])

        lines.append([0., 0., 1.])

        lines.append([1., 1., 1.])
        lines.append([-1., 1., 1.])
        lines.append([1., -1., 1.])
        lines.append([-1., -1., 1.])

        return np.array(lines)

    def point_to_line_distance(self, line1, line2, point):
        """
        See http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html.
        """
        distance_squared = ((np.square(np.linalg.norm(line1 - point)) * np.square(np.linalg.norm(line2 - line1))
                            - np.square(np.dot((line1 - point), (line2 - line1))))
                            / np.square(np.linalg.norm(line2 - line1)))


        return np.sqrt(distance_squared)

    def point_to_line_distances(self, line1, line2, points):
        og_minus_point = line1 - points
        n_og_minus_point = np.linalg.norm(og_minus_point, axis=1)
        n_line_vector_square = np.square(np.linalg.norm(line2 - line1))

        A = n_og_minus_point * n_line_vector_square
        B = np.square(np.dot(og_minus_point, (line2-line1).reshape((3,1)))).squeeze()
        C = n_line_vector_square

        return (A - B) / C



class CensiDescriptor(DescriptorAlgo):
    def __init__(self, scale_factor=100.):
        self.scale_factor = scale_factor

    def __repr__(self):
        return 'censi_{}'.format(self.scale_factor)

    def labels(self):
        labels = []

        for dim1 in ['x', 'y', 'z', 'a', 'b', 'c']:
            for dim2 in ['x', 'y', 'z', 'a', 'b', 'c']:
                labels.append('censi_{}{}'.format(dim1, dim2))

        return labels

    def compute(self, pair, reading_mask, reference_mask):
        points_reading = pair.points_of_reading()[reading_mask]
        points_reference = pair.points_of_reference()[reference_mask]

        icp_algo = IcpAlgorithm()
        icp_algo.use_vtk_inspector()
        icp_algo.max_point_density = 0

        censi_estimate = censi_estimate_from_points(points_reading, points_reference, pair.transform(), icp_algo)

        return self.scale_factor * censi_estimate.flatten()


class AveragePlanarityDescriptor(DescriptorAlgo):
    def __init__(self):
        pass

    def __repr__(self):
        return 'avg_planarity'

    def labels(self):
        return ['avg_planarity']

    def compute(self, pair, reading_mask, reference_mask):
        reading = pair.points_of_reading()[reading_mask]
        reference = pair.points_of_reference()[reference_mask]

        eigvals_reading = pair.eigenvalues_of_reading()[reading_mask]
        eigvals_reference = pair.eigenvalues_of_reference()[reference_mask]

        concat = np.vstack((eigvals_reading, eigvals_reference))

        if len(concat) == 0:
            return [0.0]
        else:
            planarities = (concat[:,1] - concat[:,0]) / np.linalg.norm(concat, axis=1)
            return [planarities.mean()]
