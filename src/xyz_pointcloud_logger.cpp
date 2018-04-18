
#include <fstream>
#include <iostream>

#include "pointcloud_io.hpp"
#include "xyz_pointcloud_logger.h"

namespace recova {

XyzPointcloudLogger::XyzPointcloudLogger(std::string&& output_dir) : output_dir(output_dir) {}

void XyzPointcloudLogger::log(const std::string& label, const Eigen::Matrix<float, 3, Eigen::Dynamic>& cloud) {
    log<float>(label, cloud);
}

void XyzPointcloudLogger::log(const std::string& label, const Eigen::Matrix<double, 3, Eigen::Dynamic>& cloud) {
    log<double>(label, cloud);
}

template<typename T>
void XyzPointcloudLogger::log(const std::string& label, const Eigen::Matrix<T, 3, Eigen::Dynamic>& cloud) {
    std::cerr << "Logging " << label << "." << '\n';


    std::ofstream output_stream;
    output_stream.open(output_dir + "/" + label + ".xyz");
    eigen_to_xyz<T>(cloud, output_stream);
    output_stream.close();
}

}
