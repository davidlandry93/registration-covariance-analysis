#!/usr/bin/env python3

import argparse
import json
import sys

from recova.util import eprint

def merge_statistics(dict1, dict2):
    return {}

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=str, nargs='+', help='The files containing the parts of the dataset')
    args = parser.parse_args()

    metadata = {}
    merged_data = []
    statistics = {}
    what = None
    for entry in args.inputs:
        with open(entry) as jsonfile:
            try:
                data = json.load(jsonfile)
            except JSONDecodeError:
                eprint('File {} was not json parsable'.format(entry))
                continue

        if 'what' in data:
            if not what:
                what = data['what']
            else:
                if data['what'] != what:
                    raise RuntimeError('Merged files of different type')

        metadata.update(data['metadata'])
        merged_data.extend(data['data'])

        if 'statistics' in data:
            statistics = merge_statistics(statistics, data['statistics'])

    output_dict = {
        'metadata': metadata,
        'data': merged_data
    }

    if what:
        output_dict['what'] = what

    json.dump(output_dict, sys.stdout)


if __name__ == '__main__':
    cli()
