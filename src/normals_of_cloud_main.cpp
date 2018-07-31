
#include <iostream>

#include <Eigen/Core>
#include <gflags/gflags.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>

#include "recov/pointcloud.h"
#include "recov/pointcloud_loader.h"

DEFINE_string(pointcloud, "", "Pointcloud on which we compute the normals.");
DEFINE_int32(k, 12, "Number of neighbors to use to compute the normal of a point.");

using namespace recov;

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string usage = "normals_of_cloud -pointcloud <pointcloud.pcd> -k <n_neighbors> ";
    gflags::SetUsageMessage(usage);

    if(FLAGS_pointcloud.empty()) {
        std::cout << usage << '\n';
        return 0;
    }

    PointcloudLoader loader;
    Pointcloud pointcloud = loader.load(FLAGS_pointcloud);
    auto pcl_pointcloud = pointcloud.as_pcl_pointcloud();

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(pcl_pointcloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

    ne.setKSearch(FLAGS_k);
    ne.compute(*cloud_normals);

    for(auto normal : cloud_normals->points) {
        std::cout << normal.normal_x << " " << normal.normal_y << " " << normal.normal_z << '\n';
    }

    return 0;
}
