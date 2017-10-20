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
    parser.add_argument('--n_bins', type=int, default=100)
    args = parser.parse_args()

    registrations = np.empty((len(json_data['data']),6))
    for i, registration in enumerate(json_data['data']):
        registration_result = np.array(registration['result_lie']).T
        registrations[i] = registration_result

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    span_dim_1 = np.max(registrations[:,args.dim1]) - np.min(registrations[:,args.dim1])
    span_dim_2 = np.max(registrations[:,args.dim2]) - np.min(registrations[:,args.dim2])

    n_bins_dim_1 = int(span_dim_1 / (span_dim_1 + span_dim_2) * args.n_bins)
    n_bins_dim_2 = int(span_dim_2 / (span_dim_1 + span_dim_2) * args.n_bins)

    print(registrations)
    counts, xedges, yedges, im = ax.hist2d(registrations[:,args.dim1], registrations[:,args.dim2], cmin=1, bins=args.n_bins)
    cbar = fig.colorbar(im, ax=ax)


    plt.show()
