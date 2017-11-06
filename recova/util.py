#!/usr/bin/env python3

import subprocess
import sys

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

def run_subprocess(command_string):
    return subprocess.check_output(
        command_string,
        universal_newlines=True,
        shell=True
    )
