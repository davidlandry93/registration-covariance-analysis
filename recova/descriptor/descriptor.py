
class Descriptor:
    """
    Given a description algorithm and a mask generator, compute a descriptor.

    The mask generator create subset of points on which we compute the description successively.
    For instance a mask generator could be a grid binning algorithm.
    The description algorithm takes a collection of points and outputs a description vector.
    """
    def __init__(self, mask_generator, description_algorithm):
        self.mask_generator = point_mask
        self.description_algo = description_algorithm

    def __repr__(self):
        return 'descriptor_{}_{}'.format(mask_generator.__repr__(), description_algorithm.__repr__())

    def compute(self, pair):
        """
        Compute the value of the descriptor for a pointcloud pair.
        """
        reading_masks, reference_masks = self.mask_generator.compute(pair)

        descriptors = [description_algorithm.compute(pair, reading_masks[i], reference_masks[i]) for i in range(len(reading_masks))]

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
                labels.append('{}_{}'.format(mask_label, descriptor_label))

        return labels
