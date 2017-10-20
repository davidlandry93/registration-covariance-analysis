#!/usr/bin/env python3

import argparse
import json
import sys

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=str, nargs='+', help='The files containing the parts of the dataset')
    args = parser.parse_args()

    metadata = {}
    merged_data = []
    for entry in args.inputs:
        with open(entry) as jsonfile:
            data = json.load(jsonfile)

            metadata.update(data['metadata'])
            merged_data.extend(data['data'])

    output_dict = {
        'metadata': metadata,
        'data': merged_data
    }
    json.dump(output_dict, sys.stdout)


if __name__ == '__main__':
    cli()
