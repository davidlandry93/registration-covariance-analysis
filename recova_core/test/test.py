
import numpy as np
import recova_core

points = np.array([[1.0, 0.0, 0.0],
                   [2.0, 3.0, 0.0],
                   [7., 3., 2.],
                   [1., 2., 0.],
                   [14., 14., 12.]])

bins = recova_core.grid_pointcloud_separator(points, 15, 15, 15, 3, 3, 3)

print(bins)

