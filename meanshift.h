/*
 * Interface for mean shift algorithm
 * Taken (and modified) from https://github.com/mattnedrich/MeanShift_cpp (licensed under MIT license)
 */

#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <vector>
#include <algorithm>
#include <functional>

namespace meanShift {
    typedef std::vector<double> position;
    typedef double (*kernelFunc)(double, double);

    struct Point {
        int label;
        position pos;
    };
    struct Cluster {
        position mode;
        std::vector<Point> original_points;
        std::vector<Point> shifted_points;

        std::size_t size() const {
            return original_points.size();
        }
        std::vector<int> getLabels() const {
            std::vector<int> labels;
            for (const Point& p : original_points) {
                labels.push_back(p.label);
            }
            return labels;
        }
    };

    class ClusterList : public std::vector<Cluster> {
        public:
            void sortBySize(bool descending=true);
    };

    double gaussianKernel(double, double);

    ClusterList cluster(const std::vector<Point> &, double, kernelFunc = gaussianKernel);
}

#endif //MEANSHIFT_H
