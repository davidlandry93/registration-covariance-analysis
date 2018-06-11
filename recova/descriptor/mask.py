import argparse
import collections
import json
import numpy as np

from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint, run_subprocess, transform_points


MaskPair = collections.namedtuple('MaskPair', ['reading', 'reference'])


class MaskGenerator:
    def __init__(self):
        pass

    def compute(self, pair):
        raise NotImplementedError('Mask generators must implement compute method')

    def __repr__(self):
        raise NotImplementedError('Mask generators must implement __repr__ method')

    def labels(self):
        raise NotImplementedError('Mask generators must implement labels method')


class IdentityMaskGenerator(MaskGenerator):
    def __repr__(self):
        return 'identity'

    def compute(self, pair):
        return MaskPair(np.ones((1, len(pair.points_of_reading())), dtype=bool),
                        np.ones((1, len(pair.points_of_reference())), dtype=bool))

    def labels(self):
        return ['identity']


class ReferenceOnlyMaskGenerator(MaskGenerator):
    def __repr__(self):
        return 'refonly'

    def compute(self, pair):
        return MaskPair(np.zeros((1, len(pair.points_of_reading())), dtype=bool),
                        np.ones((1, len(pair.points_of_reference())), dtype=bool))

    def labels(self):
        return ['refonly']


class ConcatMaskGenerator(MaskGenerator):
    def __init__(self, generators):
        self.generators = generators

    def __repr__(self):
        return '_'.join([repr(g) for g in self.generators])

    def compute(self, pair):
        len_reading = len(pair.points_of_reading())
        len_reference = len(pair.points_of_reference())

        reading_masks = np.ones((1, len_reading), dtype=bool)
        reference_masks = np.ones((1, len_reference), dtype=bool)

        for g in self.generators:
            reading_masks_to_add, reference_masks_to_add = g.compute(pair)

            n_masks = len(reading_masks_to_add) * len(reading_masks)
            new_reading_masks = np.empty((n_masks, len_reading), dtype=bool)
            new_reference_masks = np.empty((n_masks, len_reference), dtype=bool)
            for i in range(len(reading_masks)):
                for j in range(len(reading_masks_to_add)):
                    new_reading_masks[i * len(reading_masks_to_add) + j] = np.logical_and(reading_masks_to_add[j], reading_masks[i])
                    new_reference_masks[i * len(reference_masks_to_add) + j] = np.logical_and(reference_masks_to_add[j], reference_masks[i])

            reading_masks = new_reading_masks
            reference_masks = new_reference_masks

        return MaskPair(reading_masks, reference_masks)

    def labels(self):
        label_list = []
        for g in self.generators:
            new_label_list = []

            if not label_list:
                new_label_list = g.labels()
            else:
                for l in label_list:
                    for gl in g.labels():
                        new_label_list.append('{}_{}'.format(l, gl))

            label_list = new_label_list

        return label_list


class OrMaskGenerator(MaskGenerator):
    def __init__(self, generators):
        self.generators = generators

    def __repr__(self):
        return '_or_'.join([repr(g) for g in self.generators])

    def labels(self):
        label_list = []
        for g in self.generators:
            new_label_list = []

            if not label_list:
                new_label_list = g.labels()
            else:
                for l in label_list:
                    for gl in g.labels():
                        new_label_list.append('{}_or_{}'.format(l, gl))

            label_list = new_label_list

        return label_list

    def compute(self, pair):
        len_reading = len(pair.points_of_reading())
        len_reference = len(pair.points_of_reference())

        reading_masks = np.zeros((1, len_reading), dtype=bool)
        reference_masks = np.zeros((1, len_reference), dtype=bool)

        for g in self.generators:
            reading_masks_to_add, reference_masks_to_add = g.compute(pair)

            n_masks = len(reading_masks_to_add) * len(reading_masks)
            new_reading_masks = np.empty((n_masks, len_reading), dtype=bool)
            new_reference_masks = np.empty((n_masks, len_reference), dtype=bool)
            for i in range(len(reading_masks)):
                for j in range(len(reading_masks_to_add)):
                    new_reading_masks[i * len(reading_masks_to_add) + j] = np.logical_or(reading_masks_to_add[j], reading_masks[i])
                    new_reference_masks[i * len(reference_masks_to_add) + j] = np.logical_or(reference_masks_to_add[j], reference_masks[i])

            reading_masks = new_reading_masks
            reference_masks = new_reference_masks

        return MaskPair(reading_masks, reference_masks)


class AngleMaskGenerator(MaskGenerator):
    def __init__(self, angle_range=2*np.pi, angle_offset=0.0):
        self.angle_range = angle_range
        self.angle_offset = angle_offset

    def __repr__(self):
        return 'angle_{}_{}'.format(self.angle_range, self.angle_offset)

    def labels(self):
        return ['angle_{}_{}'.format(self.angle_range, self.angle_offset)]

    def compute(self, pair):
        reading_mask = self.angle_mask_of_cloud(pair.points_of_reading())
        reference_mask = self.angle_mask_of_cloud(pair.points_of_reference())

        return MaskPair([reading_mask], [reference_mask])

    def angle_mask_of_cloud(self, cloud):
        angles_around_z = np.arctan2(cloud[:,1], cloud[:,0]) + np.pi

        mask = np.logical_and(0. < angles_around_z - self.angle_offset, angles_around_z - self.angle_offset < self.angle_range)

        return mask




class OverlapMaskGenerator(MaskGenerator):
    """Return a mask containing only overlapping points."""

    def __init__(self, radius=0.1):
        self.radius = radius


    def __repr__(self):
        return 'overlap_{}'.format(self.radius)


    def labels(self):
        return ['overlap_{}'.format(self.radius)]


    def compute(self, pair):
        reading_mask_label = '{}_reading'.format(repr(self))
        reference_mask_label = '{}_reference'.format(repr(self))

        if reading_mask_label in pair.cache and reference_mask_label in pair.cache:
            return MaskPair(pair.cache[reading_mask_label], pair.cache[reference_mask_label])
        else:
            reading_array, reference_array = self._compute(pair)
            reading_array = np.array(reading_array, dtype=bool)
            reference_array = np.array(reference_array, dtype=bool)

            pair.cache[reading_mask_label] = reading_array
            pair.cache[reference_mask_label] = reference_array

            return MaskPair(reading_array, reference_array)


    def _compute(self, pair):
        cmd_template = 'overlapping_region -radius {} -mask'
        cmd_string = cmd_template.format(self.radius)

        reading = pair.points_of_reading()
        reference = pair.points_of_reference()

        input_dict = {
            'reading': reading.tolist(),
            'reference': reference.tolist(),
            't': pair.transform().tolist()
        }

        response = run_subprocess(cmd_string, json.dumps(input_dict))
        response = json.loads(response)

        reading_mask = [response['reading']]
        reference_mask = [response['reference']]

        return reading_mask, reference_mask

class CylinderGridMask(MaskGenerator):
    def __init__(self, spanr=20., spantheta=2 * np.pi, spanz=5., nr=3, ntheta=3, nz=3):
        self.span = (spanr, spantheta, spanz)
        self.n = (nr, ntheta, nz)

    def __repr__(self):
        return 'cylinder_{}_{}'.format(self.span, self.n)

    def labels(self):
        labels = []
        for i in range(self.n[0]):
            for j in range(self.n[1]):
                for k in range(self.n[2]):
                    labels.append('cylindergrid_r{}on{}_theta{}on{}_z{}on{}'.format(
                        i, self.n[0],
                        j, self.n[1],
                        k, self.n[2]
                    ))

        return labels


    def compute_for_cloud(self, cloud):
        n_points = len(cloud)
        n_masks = self.n[0] * self.n[1] * self.n[2]

        masks = np.zeros((n_masks, n_points), dtype=np.bool)

        cylindrical_cloud = np.empty(cloud.shape)
        cylindrical_cloud[:,0] = np.linalg.norm(cloud[:, 0:2], axis=1)
        cylindrical_cloud[:,1] = np.arctan2(cloud[:,1], cloud[:,0])
        cylindrical_cloud[:,2] = cloud[:,2]

        delta_r = self.span[0] / self.n[0]
        delta_theta = self.span[1] / self.n[1]
        delta_z = self.span[2] / self.n[2]

        bins = np.empty(cylindrical_cloud.shape)
        bins[:,0] = (cylindrical_cloud[:,0] + (self.span[0] / 2.)) // delta_r
        bins[:,1] = (cylindrical_cloud[:,1] + (self.span[1] / 2.)) // delta_theta
        bins[:,2] = (cylindrical_cloud[:,2] + (self.span[2] / 2.)) // delta_z

        bins = bins.astype(np.int)

        for i, point in enumerate(cylindrical_cloud):
            r, theta, z = point
            bin = bins[i]

            if (bin[0] >= self.n[0] or
                bin[0] < 0 or
                bin[1] >= self.n[1] or
                bin[1] < 0 or
                bin[2] >= self.n[2] or
                bin[2] < 0):
                continue
            else:
                masks[bin[0] * self.n[1] * self.n[2] + bin[1] * self.n[2] + bin[2]][i] = True

        return masks

    def compute(self, pair):
        transformed_ref = transform_points(pair.points_of_reference(), np.linalg.inv(pair.transform()))
        masks_reference = self.compute_for_cloud(transformed_ref)

        masks_reading = self.compute_for_cloud(pair.points_of_reading())

        return MaskPair(masks_reading, masks_reference)



class GridMaskGenerator(MaskGenerator):
    def __init__(self, spanx=10., spany=10., spanz=5., nx=3, ny=3, nz=3):
        self.span = (spanx, spany, spanz)
        self.n = (nx, ny, nz)

    def __repr__(self):
        return 'grid_{}_{}'.format(self.span, self.n)

    def labels(self):
        labels = []

        for i in range(self.n[0]):
            for j in range(self.n[1]):
                for k in range(self.n[2]):
                    labels.append('grid_x{}on{}_y{}on{}_z{}on{}'.format(
                        i, self.n[0],
                        j, self.n[1],
                        k, self.n[2]
                    ))

        return labels


    def compute(self, pair):
        linspaces = []
        for i in range(3):
            linspaces.append(np.linspace(-self.span[i] / 2.0, self.span[i] / 2.0, num=self.n[i] + 1, endpoint=True))

        reference = pair.points_of_reference()
        reading = pair.points_of_reading()

        reference = transform_points(reference, np.linalg.inv(pair.transform()))

        n_of_bins = self.n[0] * self.n[1] * self.n[2]

        reading_masks = np.empty((n_of_bins, len(reading)), dtype=bool)
        reference_masks = np.empty((n_of_bins, len(reference)), dtype=bool)

        for i in range(self.n[0]):
            for j in range(self.n[1]):
                for k in range(self.n[2]):
                    id_of_mask = i * (self.n[1] * self.n[2]) + j * self.n[2] + k

                    reading_mask = reading[:,0] > linspaces[0][i]
                    reading_mask = np.logical_and(reading_mask, reading[:,0] <= linspaces[0][i+1])
                    reading_mask = np.logical_and(reading_mask, reading[:,1] > linspaces[1][j])
                    reading_mask = np.logical_and(reading_mask, reading[:,1] <= linspaces[1][j+1])
                    reading_mask = np.logical_and(reading_mask, reading[:,2] > linspaces[2][k])
                    reading_mask = np.logical_and(reading_mask, reading[:,2] <= linspaces[2][k+1])

                    reading_masks[id_of_mask] = reading_mask

                    reference_mask = reference[:,0] > linspaces[0][i]
                    reference_mask = np.logical_and(reference_mask, reference[:,0] <= linspaces[0][i+1])
                    reference_mask = np.logical_and(reference_mask, reference[:,1] > linspaces[1][j])
                    reference_mask = np.logical_and(reference_mask, reference[:,1] <= linspaces[1][j+1])
                    reference_mask = np.logical_and(reference_mask, reference[:,2] > linspaces[2][k])
                    reference_mask = np.logical_and(reference_mask, reference[:,2] <= linspaces[2][k+1])

                    reference_masks[id_of_mask] = reference_mask

        return MaskPair(reading_masks, reference_masks)

