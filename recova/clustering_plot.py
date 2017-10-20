#!/usr/bin/env/python3

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from pylie import se3_log

def lie_vectors_of_registrations(json_data):
    lie_results = np.empty((len(json_data['data']), 6))
    for i, registration in enumerate(json_data['data']):
        m = np.array(registration['result'])
        lie_results[i,:] = se3_log(m)

    return lie_results

if __name__ == '__main__':
    json_data = json.load(sys.stdin)

    parser = argparse.ArgumentParser()
    parser.add_argument('dim1', type=int)
    parser.add_argument('dim2', type=int)
    args = parser.parse_args()

    lie_vectors = lie_vectors_of_registrations(json_data)

    ax = plt.subplot(111)

    for cluster in json_data['statistics']['clustering']:
        ax.scatter(lie_vectors[cluster, args.dim1], lie_vectors[cluster, args.dim2], s=0.5)

    ax.scatter(lie_vectors[json_data['statistics']['outliers'], args.dim1],
               lie_vectors[json_data['statistics']['outliers'], args.dim2], color='0.6', s=0.5)

    plt.show()
