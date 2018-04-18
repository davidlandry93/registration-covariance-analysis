#ifndef RECOVA_POINTCLOUD_LOGGER_H
#define RECOVA_POINTCLOUD_LOGGER_H

#include <string>

#include <Eigen/Core>

namespace recova {

class PointcloudLogger {
  public:
    virtual ~PointcloudLogger()=default;
    // virtual void log(const std::string& label, const Eigen::Matrix<float, 3, Eigen::Dynamic>& cloud)=0;
    virtual void log(const std::string& label, const Eigen::Matrix<double, 3, Eigen::Dynamic>& cloud)=0;
  private:
};

}
#endif
