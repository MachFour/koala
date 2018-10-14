/*
 * Interface for mean shift algorithm
 * Taken (and modified) from https://github.com/mattnedrich/MeanShift_cpp (licensed under MIT license)
 */

#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include "ccomponent.h"

#include <vector>
#include <algorithm>
#include <functional>

namespace meanShift {
    typedef std::vector<double> position;
    typedef double (*kernelFunc)(double, double);
    double gaussianKernel(double, double);

    template<typename T>
    struct Point {
        T data;
        position pos;
    };

    template<typename T>
    class Cluster {
    public:
        Cluster(position mode) : mode(mode), size(0) {};

        void addPoint(const Point<T> point, const Point<T> shifted) {
            original_points.push_back(point);
            shifted_points.push_back(shifted);
            size++;
        }

        // return copy
        position getMode() const {
            return position(mode);
        }

        size_t getSize() const {
            return size;
        }

        std::vector<T> getData() const {
            std::vector<T> getData;
            for (const Point<T> &p : original_points) {
                getData.push_back(p.data);
            }
            return getData;
        }

    private:
        position mode;
        size_t size;
        std::vector<Point<T>> original_points;
        std::vector<Point<T>> shifted_points;
    };

    template <typename T>
    std::vector<Cluster<T>> cluster(const std::vector<Point<T>>&, double, kernelFunc = gaussianKernel);

}

using ccCluster = meanShift::Cluster<CC>;

#endif //MEANSHIFT_H
