#ifndef RECOVA_POINTCLOUD_SEPARATOR_H
#define RECOVA_POINTCLOUD_SEPARATOR_H

#include <memory>

#include <Eigen/Core>

#include "sparse_bins.hpp"

namespace recova {

  class PointcloudSeparator {
  public:
    virtual ~PointcloudSeparator()=default;
    void set_pointcloud(std::unique_ptr<Eigen::MatrixXd>&& pointcloud);
    virtual SparseBins<Eigen::Vector3d, 3> separate() const=0;

    static Eigen::Vector3d compute_centroid(const Eigen::MatrixXd& pointcloud);

  protected:
    std::unique_ptr<Eigen::MatrixXd> pointcloud;
  };

}

#endif
