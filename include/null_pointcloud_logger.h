#ifndef RECOVA_NULL_POINTCLOUD_LOGGER
#define RECOVA_NULL_POINTCLOUD_LOGGER

#include "pointcloud_logger.h"

namespace recova {

class NullPointcloudLogger : public PointcloudLogger {
  public:
    void log(const std::string& label, const Eigen::Matrix<float, Eigen::Dynamic, 3>& cloud) {}
    void log(const std::string& label, const Eigen::Matrix<double, Eigen::Dynamic, 3>& cloud) {}
  private:
};

}

#endif
