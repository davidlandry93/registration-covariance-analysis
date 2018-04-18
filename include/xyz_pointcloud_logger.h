#ifndef RECOVA_XYZ_POINTCLOUD_LOGGER_H
#define RECOVA_XYZ_POINTCLOUD_LOGGER_H

#include <string>

#include "pointcloud_logger.h"

namespace recova {

class XyzPointcloudLogger : public PointcloudLogger {
  public:
    XyzPointcloudLogger(std::string& output_dir);

    void log(const std::string& label, const Eigen::Matrix<float, Eigen::Dynamic, 3>& cloud);
    void log(const std::string& label, const Eigen::Matrix<double, Eigen::Dynamic, 3>& cloud);
  private:
    std::string output_dir;

    template<typename T>
    void log(const std::string& label, const Eigen::Matrix<T, Eigen::Dynamic, 3>& cloud);
};

}


#endif
