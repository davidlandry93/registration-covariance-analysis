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
    usage += "-pcd <path.pcd> [-qpc <path.qpc>]";
    gflags::SetUsageMessage(usage);

    if(FLAGS_pointcloud.empty()) {
        std::cout << usage << '\n';
        return 0;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if(pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_pointcloud, *cloud) == -1) {
        throw std::runtime_error("Error while loading the pcd file.");
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3> points(cloud->points.size(), 3);
    for (auto i = 0; i < cloud->points.size(); ++i) {
        points(i,0) = cloud->points[i].x;
        points(i,1) = cloud->points[i].y;
        points(i,2) = cloud->points[i].z;
    }

    if(FLAGS_output.empty() || FLAGS_output == "-") {
        recov::write_qpc(points, std::cout);
    } else {
        std::ofstream outstream;
        outstream.open(FLAGS_output, std::ios::binary);
        recov::write_qpc(points, outstream);
    }

    return 0;
}
