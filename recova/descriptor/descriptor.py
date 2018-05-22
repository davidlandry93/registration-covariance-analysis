
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
        return 'descriptor_{}_{}'.format(self.mask_generator.__repr__(),
                                         self.description_algo.__repr__())

    def _compute(self, pair):
        """
        Compute the value of the descriptor for a pointcloud pair.
        """
        reading_masks, reference_masks = self.mask_generator.compute(pair)

        descriptors = []
        for i in range(len(reading_masks)):
            descriptor = self.description_algo.compute(pair, reading_masks[i], reference_masks[i])
            descriptors.append(descriptor)

        flattened_descriptor = []
        for l in descriptors:
            for element in l:
                flattened_descriptor.append(element)

        return flattened_descriptor

    def compute(self, pair):
        cached_descriptor = pair.cache[repr(self)]

        if cached_descriptor is None:
            descriptor = self._compute(pair)
            pair.cache[repr(self)] = descriptor
            return np.array(descriptor)

        else:
            return np.array(cached_descriptor)


    def labels(self):
        mask_labels = self.mask_generator.labels()
        descriptor_labels = self.description_algo.labels()

        labels = []
        for mask_label in mask_labels:
            for descriptor_label in descriptor_labels:
                labels.append('{}___{}'.format(mask_label, descriptor_label))

        return labels



class DescriptorConcat(Descriptor):
    def __init__(self, descriptors):
        self.descriptors = descriptors

    def __repr__(self):
        desc_reprs = [repr(x) for x in self.descriptors]

        return '_'.join(desc_reprs)

    def labels(self):
        labels = []
        for descriptor in self.descriptors:
            labels.extend(descriptor.labels())

        return labels

    def compute(self, pair):
        descr_concat = []
        for descriptor in self.descriptors:
            descr_concat.extend(descriptor.compute(pair))

        return np.array(descr_concat)



