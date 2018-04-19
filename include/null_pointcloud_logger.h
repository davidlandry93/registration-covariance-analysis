#ifndef RECOVA_NULL_POINTCLOUD_LOGGER
#define RECOVA_NULL_POINTCLOUD_LOGGER

#include "pointcloud_logger.h"

namespace recova {

class NullPointcloudLogger : public PointcloudLogger {
  public:
    void log(const std::string& label, const Eigen::Matrix<float, 3, Eigen::Dynamic>& cloud) const {}
    void log(const std::string& label, const Eigen::Matrix<double, 3, Eigen::Dynamic>& cloud) const {}
    bool enabled() const { return false; }
  private:
};

}

#endif
