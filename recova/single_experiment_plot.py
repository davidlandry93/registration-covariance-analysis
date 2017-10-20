#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == '__main__':
    json_data = json.load(sys.stdin)

    parser = argparse.ArgumentParser()
    parser.add_argument('dim1', type=int)
    parser.add_argument('dim2', type=int)
    args = parser.parse_args()


    registrations = np.empty((len(json_data['data']),6))
    for i, registration in enumerate(json_data['data']):
        registration_result = np.array(registration['result_lie']).T
        registrations[i] = registration_result

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(registrations[:,args.dim1], registrations[:,args.dim2], s=0.5)
    plt.axis('equal')
    plt.show()
