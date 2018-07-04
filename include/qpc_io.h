
#include <Eigen/Core>

namespace recov {

Eigen::Matrix<double, Eigen::Dynamic, 3> read_qpc(std::istream& instream);
void write_qpc(const Eigen::Matrix<double, Eigen::Dynamic, 3>& pointcloud, std::ostream& outstream);

}
