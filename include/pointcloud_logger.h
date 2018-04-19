#ifndef RECOVA_POINTCLOUD_LOGGER_H
#define RECOVA_POINTCLOUD_LOGGER_H

#include <string>

#include <Eigen/Core>

namespace recova {

class PointcloudLogger {
  public:
    virtual ~PointcloudLogger()=default;
    // virtual void log(const std::string& label, const Eigen::Matrix<float, 3, Eigen::Dynamic>& cloud)=0;
    virtual void log(const std::string& label, const Eigen::Matrix<double, 3, Eigen::Dynamic>& cloud) const=0;
    virtual void log_6d(const std::string& label, const Eigen::Matrix<double,6,Eigen::Dynamic>& cloud) const;

    virtual bool enabled() const=0;
  private:
};

}
#endif
