import argparse
import collections
import json
import numpy as np

from recova.registration_dataset import points_to_vtk
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint, run_subprocess


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


class OverlapMaskGenerator(MaskGenerator):
    """Return a mask containing only overlapping points."""

    def __init__(self, radius=0.1):
        self.radius = radius


    def __repr__(self):
        return 'overlap_{}'.format(self.radius)


    def labels(self):
        return ['overlap_{}'.format(self.radius)]


    def compute(self, pair):
        cached_reading_entry = pair.cache['{}_reading'.format(repr(self))]
        cached_reference_entry = pair.cache['{}_reference'.format(repr(self))]

        if cached_reading_entry  is not None and cached_reference_entry is not None:
            return MaskPair(np.array(cached_reading_entry, dtype=bool), np.array(cached_reference_entry, dtype=bool))
        else:
            reading_array, reference_array = self._compute(pair)
            pair.cache['{}_reading'.format(repr(self))] = reading_array
            pair.cache['{}_reference'.format(repr(self))] = reference_array
            return MaskPair(np.array(reading_array, dtype=bool), np.array(reference_array, dtype=bool))


    def _compute(self, pair):
        cmd_template = 'overlapping_region -radius {} -mask'
        cmd_string = cmd_template.format(self.radius)

        reading = pair.points_of_reading()
        reference = pair.points_of_reference()

        eprint('Len reading: {}'.format(len(reading)))
        eprint('Len reference: {}'.format(len(reference)))

        input_dict = {
            'reading': reading.tolist(),
            'reference': reference.tolist(),
            't': pair.ground_truth().tolist()
        }

        response = run_subprocess(cmd_string, json.dumps(input_dict))
        response = json.loads(response)

        reading_mask = [response['reading']]
        reference_mask = [response['reference']]

        return reading_mask, reference_mask



class GridMaskGenerator(MaskGenerator):
    def __init__(self, spanx=10., spany=10., spanz=10., nx=3, ny=3, nz=3):
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

def mask_generator_factory(mask_name):
    if mask_name == 'grid':
        return GridMaskGenerator()
    elif mask_name == 'overlap':
        return OverlapMaskGenerator()
    elif mask_name == 'identity':
        return IdentityMaskGenerator()
    else:
        raise ValueError('Unknown mask generator {}'.format(mask_name))


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='Location of the registration result database to use.')
    parser.add_argument('dataset', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('mask', type=str)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.dataset, args.reading, args.reference)

    reading = pair.points_of_reading()
    reference = pair.points_of_reference()

    mask_generator = mask_generator_factory(args.mask)

    reading_masks, reference_masks = mask_generator.compute(pair)

    print(reading.shape)
    print(reading_masks.shape)
    print(reading[reading_masks[0]])

    for i in range(len(reading_masks)):
        if reading_masks[i].any():
            points_to_vtk(reading[reading_masks[i]], '{}_reading_{}'.format(mask_generator.__repr__(), i))
        if reference_masks[i].any():
            points_to_vtk(reference[reference_masks[i]], '{}_reference_{}'.format(mask_generator.__repr__(), i))
