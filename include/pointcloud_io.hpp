#ifndef RECOVA_POINTCLOUD_IO_HPP
#define RECOVA_POINTCLOUD_IO_HPP

#include "pointcloud_io.h"

namespace recova {

template <typename T>
void eigen_to_xyz(const Eigen::Matrix<T, 3, Eigen::Dynamic>& m,
                  std::ostream& os) {
    os.precision(10);

    for(auto i = 0; i < m.cols(); i++) {
        os << m(0,i) << " " << m(1,i) << " " << m(2,i) << '\n';
    }
}

}



#endif
