import numpy as np

class DescriptorAlgorithm:
    """A descriptor algorithm takes a bin of points and outputs a description of the contents of the bin."""
    def __init__(self):
        pass

    def compute(self, pointcloud, bins):
        raise NotImplementedError('DescriptorAlgorithms must implement method compute')

    def __repr__(self):
        raise NotImplementedError('DescriptorAlgorithms must implement __repr__')


class OccupancyGridDescriptor(DescriptorAlgorithm):
    def compute(self, pointcloud, bins):
        descriptor = [len(x) / len(pointcloud) for x in bins]
        return descriptor

    def __repr__(self):
        return 'occupancy_grid'


class MomentGridDescriptor(DescriptorAlgorithm):
    def compute(self, pointcloud, bins):
        bin_descriptors = np.empty((len(bins), 12))
        for i, b in enumerate(bins):
            if len(b) == 0:
                bin_descriptors[i] = np.zeros(12)
            else:
                points = np.array(b)
                first_moment = points.mean(axis=0)

                centered_points = points - first_moment
                second_moment = np.dot(centered_points.T, centered_points) / len(centered_points)

                bin_descriptors[i, 0:3] = first_moment
                bin_descriptors[i, 3:12] = second_moment.flatten()

        return bin_descriptors.flatten().tolist()


    def __repr__(self):
        return 'moment-grid'
