
#include <cmath>
#include <Eigen/Core>

#include "grid_pointcloud_separator.h"
#include "sparse_bins.hpp"

namespace recova {
  GridPointcloudSeparator::GridPointcloudSeparator(double spanx, double spany, double spanz,
                                                   int nx, int ny, int nz) :
    spanx(spanx), spany(spany), spanz(spanz), nx(nx), ny(ny), nz(nz) {}

  SparseBins<Eigen::MatrixXd, 3> GridPointcloudSeparator::separate() const {
    auto centroid = compute_centroid(*pointcloud);
    auto bins = SparseBins<Eigen::MatrixXd, 3>(Eigen::MatrixXd(3,0));

    double deltax = spanx / (double) nx;
    double deltay = spany / (double) ny;
    double deltaz = spanz / (double) nz;

    for(auto i = 0; i < pointcloud->cols(); i++) {
      Eigen::Vector3d point = pointcloud->col(i);
      auto centered_point = point - centroid;

      int x = static_cast<int>(std::floor(centered_point[0] + 0.5 * spanx / deltax));
      int y = static_cast<int>(std::floor(centered_point[1] + 0.5 * spany / deltay));
      int z = static_cast<int>(std::floor(centered_point[2] + 0.5 * spanz / deltaz));

      auto current_bin = bins.get({x,y,z});
      current_bin.conservativeResize(current_bin.rows(), current_bin.cols()+1);
      current_bin.col(current_bin.cols()-1) = point;
    }

    return bins;
  }

  Eigen::Vector3d compute_centroid(const Eigen::MatrixXd& pointcloud) {
    Eigen::Vector3d sum(0., 0., 0.);
    for(auto i = 0; i < pointcloud.cols(); i++) {
      sum += pointcloud.col(i);
    }

    return sum / pointcloud.cols();
  }
}
