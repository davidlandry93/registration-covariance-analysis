
import json
import numpy as np

from recova.util import eprint, run_subprocess

class PointcloudCombiner:
    """A pointcloud combiner takes the reading and outputs a single pointcloud which we use for learning."""
    def compute(self, reading, reference, t):
        raise NotImplementedError('PointcloudCombiners must implement compute')

    def __repr__(self):
        raise NotImplementedError('PoincloudCombiners must implement __repr__')



class ReferenceOnlyCombiner(PointcloudCombiner):
    def compute(self, reading, reference, t):
        return reference

    def __repr__(self):
        return 'ref-only'


class OverlappingRegionCombiner(PointcloudCombiner):
    """A pointcloud combiner that registers the point clouds and returns the overlapping region."""
    def compute(self, reading, reference, ground_truth):
        cmd_template = 'overlapping_region'

        input_dict = {
            'reading': reading.tolist(),
            'reference': reference.tolist(),
            't': ground_truth.tolist()
        }

        response = run_subprocess(cmd_template, json.dumps(input_dict))

        response = json.loads(response)
        reading_points = np.array(response['reading'])
        reference_points = np.array(response['reference'])

        return np.vstack((reading_points, reference_points))

    def __repr__(self):
        return 'overlapping-region_gt'
