
import argparse
import json
import time

from recov.pointcloud_io import pointcloud_to_vtk
from recova.descriptor.algo import CensiDescriptor, ConcatDescriptorAlgo, MomentsDescriptorAlgo, NormalsHistogramDescriptionAlgo, OccupancyDescriptorAlgo, AveragePlanarityDescriptor
from recova.descriptor.descriptor import Descriptor, DescriptorConcat 
from recova.descriptor.mask import AngleMaskGenerator, ConcatMaskGenerator, IdentityMaskGenerator, CylinderGridMask, GridMaskGenerator, OverlapMaskGenerator, OrMaskGenerator, ReferenceOnlyMaskGenerator
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint, transform_points


def apply_params_to_instance(instance, params):
    for key in params:
        setattr(instance, key, params[key])

    return instance


def single_description_algo_factory(config):
    if config['name'] == 'moments':
        instance = MomentsDescriptorAlgo()
    elif config['name'] == 'normals_histogram':
        instance = NormalsHistogramDescriptionAlgo()
    elif config['name'] == 'occupancy':
        instance = OccupancyDescriptorAlgo()
    elif config['name'] == 'censi':
        instance = CensiDescriptor()
    elif config['name'] == 'planarity_avg':
        instance = AveragePlanarityDescriptor()
    else:
        raise ValueError('No description algo named {}'.format(config['name']))

    conf_copy = dict(config)
    del conf_copy['name']
    return apply_params_to_instance(instance, conf_copy)


def description_algo_factory(config):
    if isinstance(config, list):
        return ConcatDescriptorAlgo([description_algo_factory(x) for x in config])
    else:
        return single_description_algo_factory(config)


def single_mask_factory(config):
    if config['name'] == 'identity':
        instance = IdentityMaskGenerator()
    elif config['name'] == 'grid':
        instance = GridMaskGenerator()
    elif config['name'] == 'overlap':
        instance = OverlapMaskGenerator()
    elif config['name'] == 'cylinder':
        instance = CylinderGridMask()
    elif config['name'] == 'angle':
        instance = AngleMaskGenerator()
    elif config['name'] == 'reference-only':
        instance = ReferenceOnlyMaskGenerator()
    else:
        raise ValueError('No mask generator named {}'.format(config['name']))

    conf_copy = dict(config)
    del conf_copy['name']
    return apply_params_to_instance(instance, conf_copy)


def mask_factory(config):
    if isinstance(config, list):
        return ConcatMaskGenerator([mask_factory(x) for x in config])
    elif isinstance(config, dict) and 'name' in config and config['name'] == 'or':
        return OrMaskGenerator([mask_factory(x) for x in config['masks']])
    else:
        return single_mask_factory(config)


def single_descriptor_factory(config):
    return Descriptor(mask_factory(config['mask']), description_algo_factory(config['algo']))

def descriptor_factory(config):
    if isinstance(config, list):
        return DescriptorConcat([descriptor_factory(x) for x in config])
    else:
        return single_descriptor_factory(config)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='Location of the registration result database to use.')
    parser.add_argument('dataset', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('-c', '--config', type=str, help='Path to a json file containing the descriptor configuration.')
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.dataset, args.reading, args.reference)

    if args.config:
        with open(args.config) as f:
            descriptor = descriptor_factory(json.load(f))
    else:
        config = {'mask': {'name': 'grid'},
                  'algo': {'name': 'normals_histogram'}}
        descriptor = descriptor_factory(config)

    descriptor_compute_start = time.time()
    computed_descriptor = descriptor.compute(pair)
    eprint('Descriptor took {} seconds'.format(time.time() - descriptor_compute_start))

    print(computed_descriptor)
    print(descriptor.labels())




def apply_mask_cli():
    parser = argparse.ArgumentParser(description='Apply as point selection mask on a pair of pointclouds.')
    parser.add_argument('database', type=str, help='Location of the registration result database to use.')
    parser.add_argument('dataset', type=str)
    parser.add_argument('reading', type=int)
    parser.add_argument('reference', type=int)
    parser.add_argument('--output', type=str, default='.', help='Output directory of the visualization.')
    parser.add_argument('--radius', type=float, default=0.1, help='For the overlap mask generator, the max distance between points for them to be neighbors.')
    parser.add_argument('--range', type=float, help='For the angle mask generator, the range of angles accepted.', default=0.0)
    parser.add_argument('--offset', type=float, help='For the angle mask generator, the offset of angles accepted.', default=0.0)
    parser.add_argument('-c', '--config', type=str, help='Path to a json config for the mask')
    parser.add_argument('-r', '--rotation', type=float, help='Rotation around the z axis to apply to the cloud pair before computing the descriptor, in radians.', default=0.0)
    args = parser.parse_args()

    db = RegistrationPairDatabase(args.database)
    pair = db.get_registration_pair(args.dataset, args.reading, args.reference)

    pair.rotation_around_z = args.rotation

    reading = pair.points_of_reading()
    reference = pair.points_of_reference()

    with open(args.config) as f:
        mask_generator = mask_factory(json.load(f))

    reading_masks, reference_masks = mask_generator.compute(pair)


    eprint(mask_generator.labels())


    pointcloud_to_vtk(reference, args.output + '/reference')
    pointcloud_to_vtk(transform_points(reading, pair.transform()), args.output + '/reading')


    for i in range(len(reading_masks)):
        if reference_masks[i].any():
            pointcloud_to_vtk(reference[reference_masks[i]], args.output + '/' + '{}_reference_{}'.format(mask_generator.__repr__(), i))

        if reading_masks[i].any():
            transformed_masked_reading= transform_points(reading[reading_masks[i]], pair.transform())

            pointcloud_to_vtk(transformed_masked_reading, args.output + '/' + '{}_reading_{}'.format(mask_generator.__repr__(), i))

if __name__ == '__main__':
    cli()
