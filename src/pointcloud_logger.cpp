
#include "pointcloud_logger.h"

namespace recova {

void PointcloudLogger::log_6d(const std::string& label, const Eigen::Matrix<double,6,Eigen::Dynamic>& points) const {
    log(label + "_translation", points.topRows(3));
    log(label + "_rotation", points.bottomRows(3));
}

}
