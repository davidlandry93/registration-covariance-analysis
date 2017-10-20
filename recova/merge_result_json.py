#!/usr/bin/env python3

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, help='The file where to output the merged datasets')
    parser.add_argument('--inputs', type=str, nargs='+', help='The files containing the parts of the dataset')

    args = parser.parse_args()

    metadata = {}
    merged_data = []
    for entry in args.inputs:
        with open(entry) as jsonfile:
            data = json.load(jsonfile)

            metadata.update(data['metadata'])
            merged_data.extend(data['data'])

    with open(args.output, 'w') as jsonfile:
        output_dict = {
            'metadata': metadata,
            'data': merged_data
        }
        json.dump(output_dict, jsonfile)



