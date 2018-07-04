
#include <stdexcept>

#include "qpc_io.h"

namespace recov {

Eigen::Matrix<double, Eigen::Dynamic, 3> read_qpc(std::istream& instream) {
    // Read qpc version
    char qpc_version;
    instream.read(&qpc_version, sizeof(char));

    if(qpc_version != 0) {
        throw std::runtime_error("Unhandled QPC file version.");
    }

    int n_points;
    instream.read((char*)&n_points, sizeof(int));

    Eigen::Matrix<double, Eigen::Dynamic, 3> points(n_points, 3);
    for(auto i = 0; i < n_points; i++) {
        double coords[3];

        instream.read((char*) coords, 3*sizeof(double));
        points(i, 0) = coords[0];
        points(i, 1) = coords[1];
        points(i, 2) = coords[2];
    }

    return points;
}

void write_qpc(const Eigen::Matrix<double, Eigen::Dynamic, 3>& points, std::ostream& outstream) {
    char version = 0;
    outstream.write(&version, sizeof(char));

    int n_points = points.rows();
    outstream.write((char*) &n_points, sizeof(int));

    for (auto i = 0; i < n_points; i++) {
        double coords[3] = {points(i,0), points(i,1), points(i,2)};
        outstream.write((char*) coords, 3*sizeof(double));
    }
}

}
