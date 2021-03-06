#!/usr/bin/env python3

import json
import numpy as np
import subprocess


def density_of_points(points, k=12):
    command = 'point_density -k {}'.format(k)

    response = subprocess.run(command,
                              input=json.dumps(points.tolist()),
                              stdout=subprocess.PIPE,
                              shell=True,
                              universal_newlines=True)

    return np.array(json.loads(response.stdout))
