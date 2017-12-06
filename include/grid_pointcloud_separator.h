#ifndef RECOVA_GRID_POINTCLOUD_SEPARATOR_H
#define RECOVA_GRID_POINTCLOUD_SEPARATOR_H

#include <Eigen/Core>

#include "pointcloud_separator.h"

namespace recova {
  class GridPointcloudSeparator : public PointcloudSeparator {
  public:
    GridPointcloudSeparator(double spanx, double spany, double spanz, int nx, int ny, int nz);
    SparseBins<Eigen::Vector3d, 3> separate() const override;

  private:
    double spanx, spany, spanz;
    int nx, ny, nz;
  };
}

#endif
