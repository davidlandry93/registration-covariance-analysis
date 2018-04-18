#ifndef RECOVA_POINTCLOUD_LOGGER_H
#define RECOVA_POINTCLOUD_LOGGER_H

#include <string>

#include <Eigen/Core>

namespace recova {

class PointcloudLogger {
  public:
    virtual ~PointcloudLogger()=default;
    virtual void log(const std::string& label, const Eigen::Matrix<float, Eigen::Dynamic, 3>& cloud)=0;
    virtual void log(const std::string& label, const Eigen::Matrix<double, Eigen::Dynamic, 3>& cloud)=0;
  private:
};

}
#endif
