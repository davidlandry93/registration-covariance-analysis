#!/usr/bin/env python3

import json
import numpy as np
import sys

if __name__ == '__main__':
    dataset = json.load(sys.stdin)
    ground_truth_inv = np.matrix(dataset['metadata']['ground_truth']).I

    for i, result in enumerate(dataset['data']):
        result_matrix = np.matrix(result['result'])
        estimate_matrix = np.matrix(result['initial_estimate'])

        centered_result = ground_truth_inv * result_matrix
        centered_estimate = ground_truth_inv * estimate_matrix

        dataset['data'][i]['result'] = centered_result.tolist()
        dataset['data'][i]['initial_estimate'] = centered_estimate.tolist()

    json.dump(dataset, sys.stdout)
