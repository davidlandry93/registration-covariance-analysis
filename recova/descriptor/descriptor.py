
import argparse
import numpy as np
import recova.descriptor.mask
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint
import time

class Descriptor:
    """
    Given a description algorithm and a mask generator, compute a descriptor.

    The mask generator create subset of points on which we compute the description successively.
    For instance a mask generator could be a grid binning algorithm.
    The description algorithm takes a collection of points and outputs a description vector.
    """
    def __init__(self, mask_generator, description_algorithm):
        self.mask_generator = mask_generator
        self.description_algo = description_algorithm

    def __repr__(self):
        return 'descriptor_{}_{}'.format(mask_generator.__repr__(), description_algorithm.__repr__())

    def compute(self, pair):
        """
        Compute the value of the descriptor for a pointcloud pair.
        """
        reading_masks, reference_masks = self.mask_generator.compute(pair)

        descriptors = [self.description_algo.compute(pair, reading_masks[i], reference_masks[i]) for i in range(len(reading_masks))]

        flattened_descriptor = []
        for l in descriptors:
            for element in l:
                flattened_descriptor.append(element)

        return np.array(flattened_descriptor)

    def labels(self):
        mask_labels = self.mask_generator.labels()
        descriptor_labels = self.description_algo.labels()

        labels = []
        for mask_label in mask_labels:
            for descriptor_label in descriptor_labels:
                labels.append('{}___{}'.format(mask_label, descriptor_label))

        return labels


class DescriptorAlgo:
    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError('Descriptor algorithms must implement __repr__ method')

    def compute(self, pair, reading_mask, reference_mask):
        raise NotImplementedError('Descriptor algorithms must implement compute method')

    def labels(self):
        raise NotImplementedError('Descriptor algorithms must implement labels method')


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

        print(reading.shape)
        print(reference.shape)

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



class NormalHistogramDescriptionAlgo(DescriptorAlgo):
    def __init__(self):
        pass

    def __repr__(self):
        return 'norm_histogram'

    def compute(self, pair, reading_mask, reference_mask):
        normals_reading = pair.normals_of_reading()

        normals_reference = pair.normals_of_reference()[reference_mask]

        normals = np.vstack((normals_reading, normals_reference))

        ref_lines = self.reference_lines()
        histogram = np.zeros(len(ref_lines))

        distances = np.empty((len(normals), len(ref_lines)))
        for i, line in enumerate(ref_lines):
            distances[:,i] = self.point_to_line_distances(np.array([0.,0.,0.]), line, normals)

        # for normal in normals:
        #     distances = [self.point_to_line_distance(np.array([0., 0., 0.]), x, normal) for x in ref_lines]
        #     histogram[np.argmin(distances)] += 1.

        eprint(distances)

        mins = np.argmin(distances, axis=1)
        eprint(mins)

        histogram = np.zeros(len(ref_lines))

        for minimum in mins:
            histogram[minimum] += 1.

        eprint(histogram)

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





def descriptor_algo_factory(descriptor):
    if descriptor == 'moments':
        return MomentsDescriptorAlgo()
    elif descriptor == 'normal_histogram':
        return NormalHistogramDescriptionAlgo()
    else:
        raise ValueError('No descriptor called {}'.format(descriptor))


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='Location of the registration result database to use.')
    parser.add_argument('dataset', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('mask', type=str)
    parser.add_argument('descriptor', type=str)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.dataset, args.reading, args.reference)

    mask = recova.descriptor.mask.mask_generator_factory(args.mask)
    descriptor_algo = descriptor_algo_factory(args.descriptor)

    descriptor_algo = Descriptor(mask, descriptor_algo)

    descriptor_compute_start = time.time()
    descriptor = descriptor_algo.compute(pair)
    eprint('Descriptor took {} seconds'.format(time.time() - descriptor_compute_start))

    print(descriptor)
    print(descriptor_algo.labels())

    print(len(descriptor))
    print(len(descriptor_algo.labels()))


if __name__ == '__main__':
    cli()
