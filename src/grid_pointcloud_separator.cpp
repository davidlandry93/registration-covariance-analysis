
#include <cmath>
#include <Eigen/Core>
#include <iostream>

#include "grid_pointcloud_separator.h"
#include "sparse_bins.hpp"

namespace recova {
  GridPointcloudSeparator::GridPointcloudSeparator(double spanx, double spany, double spanz,
                                                   int nx, int ny, int nz) :
    spanx(spanx), spany(spany), spanz(spanz), nx(nx), ny(ny), nz(nz) {}

  SparseBins<Eigen::Vector3d, 3> GridPointcloudSeparator::separate() const {
    auto centroid = compute_centroid(*pointcloud);
    SparseBins<Eigen::Vector3d, 3> bins;

    double deltax = spanx / (double) nx;
    double deltay = spany / (double) ny;
    double deltaz = spanz / (double) nz;

    for(auto i = 0; i < pointcloud->cols(); i++) {
      Eigen::Vector3d point = pointcloud->col(i);
      auto centered_point = point - centroid;

      int x = static_cast<int>(std::floor(centered_point[0] / deltax)) + nx / 2;
      int y = static_cast<int>(std::floor(centered_point[1] / deltay)) + ny / 2;
      int z = static_cast<int>(std::floor(centered_point[2] / deltaz)) + nz / 2;

      bins.add_to_bin({x,y,z}, point);
    }

    return bins;
  }

}
