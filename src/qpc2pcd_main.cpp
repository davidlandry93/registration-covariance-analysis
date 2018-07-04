#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <gflags/gflags.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "qpc_io.h"

DEFINE_string(pointcloud, "", "Path of the pcd to convert");
DEFINE_string(output, "", "Path to the output qpc");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string usage;
    usage += argv[0];
    usage += "-pointcloud <path.qpc> -output <path.pcd>";
    gflags::SetUsageMessage(usage);

    if(FLAGS_output.empty()) {
        std::cout << usage << '\n';
        return 0;
    }

    std::ifstream instream;
    instream.open(FLAGS_pointcloud, std::ios::binary);
    auto points = recov::read_qpc(instream);
    instream.close();

    pcl::PointCloud<pcl::PointXYZ> cloud;
    for(auto i = 0; i < points.rows(); ++i) {
        pcl::PointXYZ p;
        p.x = points(i,0);
        p.y = points(i,1);
        p.z = points(i,2);
        cloud.push_back(p);
    }

    pcl::io::savePCDFileBinary(FLAGS_output, cloud);

    return 0;
}
