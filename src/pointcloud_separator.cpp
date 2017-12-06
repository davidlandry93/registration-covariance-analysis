
#include <iostream>

#include "pointcloud_separator.h"

namespace recova {
  void PointcloudSeparator::set_pointcloud(std::unique_ptr<Eigen::MatrixXd>&& p_pointcloud) {
    pointcloud = std::move(p_pointcloud);
  }

  Eigen::Vector3d PointcloudSeparator::compute_centroid(const Eigen::MatrixXd& pointcloud) {
    Eigen::Vector3d sum(0., 0., 0.);
    for(auto i = 0; i < pointcloud.cols(); i++) {
      sum += pointcloud.col(i);
    }

    auto centroid = sum / pointcloud.cols();
    return centroid;
  }
}

