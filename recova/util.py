#!/usr/bin/env python3

POSITIVE_STRINGS = ('yes', 'y', 't', 'true', '1')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def empty_to_none(dictionary):
    return dictionary if dictionary else None

def parse_dims(dim_string):
    """
    Parse a string containing a list of dimensions to use when plotting.
    """
    dims = [int(x.strip()) for x in dim_string.split(',')]
    if len(dims) != 3:
        raise RuntimeError('Can only generate an ellipsoid with 3 dimensions.')

    return dims
