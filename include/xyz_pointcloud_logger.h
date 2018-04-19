#ifndef RECOVA_XYZ_POINTCLOUD_LOGGER_H
#define RECOVA_XYZ_POINTCLOUD_LOGGER_H

#include <string>

#include "pointcloud_logger.h"

namespace recova {

class XyzPointcloudLogger : public PointcloudLogger {
  public:
    XyzPointcloudLogger(std::string&& output_dir);

    void log(const std::string& label, const Eigen::Matrix<float, 3, Eigen::Dynamic>& cloud) const;
    void log(const std::string& label, const Eigen::Matrix<double, 3, Eigen::Dynamic>& cloud) const;
    bool enabled() const;
  private:
    std::string output_dir;

    template<typename T>
    void log(const std::string& label, const Eigen::Matrix<T, 3, Eigen::Dynamic>& cloud) const;
};

}


#endif
