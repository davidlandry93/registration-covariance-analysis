
import numpy as np

def to_homogeneous(pointcloud):
    if pointcloud.shape[0] > pointcloud.shape[1]:
        pointcloud = pointcloud.T

    homogeneous_pointcloud = np.ones((4, pointcloud.shape[1]))
    homogeneous_pointcloud[0:3, 0:] = pointcloud

    return homogeneous_pointcloud
