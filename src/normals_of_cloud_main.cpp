
#include <iostream>

#include <Eigen/Core>
#include <gflags/gflags.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>

DEFINE_string(pointcloud, "", "Pointcloud on which we compute the normals.");
DEFINE_int32(k, 12, "Number of neighbors to use to compute the normal of a point.");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string usage = "normals_of_cloud -pointcloud <pointcloud.pcd> -k <n_neighbors> ";
    gflags::SetUsageMessage(usage);

    if(FLAGS_pointcloud.empty()) {
        std::cout << usage << '\n';
        return 0;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if(pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_pointcloud, *cloud) == -1) {
        std::cout << "Could not load pointcloud" << '\n';
        return 0;
    }

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

    ne.setKSearch(FLAGS_k);
    ne.compute(*cloud_normals);

    std::cerr << cloud->points.size() << " points in the cloud" << '\n';
    std::cerr << cloud_normals->points.size() << " normals in the vector" << '\n';

    for(auto normal : cloud_normals->points) {
        std::cout << normal.normal_x << " " << normal.normal_y << " " << normal.normal_z << '\n';
    }

    return 0;
}
