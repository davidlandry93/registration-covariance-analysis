
#include <fstream>

#include "pointcloud_io.hpp"
#include "xyz_pointcloud_logger.h"

namespace recova {

XyzPointcloudLogger::XyzPointcloudLogger(std::string& output_dir) : output_dir(output_dir) {}

void XyzPointcloudLogger::log(const std::string& label, const Eigen::Matrix<float, Eigen::Dynamic, 3>& cloud) {
    log<float>(label, cloud);
}

void XyzPointcloudLogger::log(const std::string& label, const Eigen::Matrix<double, Eigen::Dynamic, 3>& cloud) {
    log<double>(label, cloud);
}

template<typename T>
void XyzPointcloudLogger::log(const std::string& label, const Eigen::Matrix<T, Eigen::Dynamic, 3>& cloud) {
    std::ofstream output_stream;
    output_stream.open(output_dir + label);
    eigen_to_xyz<T>(cloud, output_stream);
    output_stream.close();
}

}
