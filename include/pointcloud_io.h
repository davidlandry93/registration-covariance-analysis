#ifndef RECOVA_POINTCLOUD_IO_H
#define RECOVA_POINTCLOUD_IO_H

#include <iostream>

#include <Eigen/Core>

namespace recova {

template <typename T>
void eigen_to_xyz(const Eigen::Matrix<T, 3, Eigen::Dynamic>& m,
                  std::ostream& os);
}

#endif
