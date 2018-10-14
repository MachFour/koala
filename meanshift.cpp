/*
 * Implementation for mean shift algorithm
 * Taken (and modified) from https://github.com/mattnedrich/MeanShift_cpp (licensed under MIT license)
 */

#include <cmath>
#include "meanshift.h"

#define CLUSTER_EPSILON 0.5

namespace meanShift {
    using std::vector;

    double euclidean_distance_sqr(const position &a, const position &b) {
        double total = 0;
        for (unsigned i = 0; i < a.size(); i++) {
            total += (a[i] - b[i])*(a[i] - b[i]);
        }
        return total;
    }

    double euclidean_distance(const position &a, const position &b) {
        return sqrt(meanShift::euclidean_distance_sqr(a, b));
    }

    // MAKE SURE bandwidth > 0
    double gaussianKernel(double distance, double bandwidth) {
        return exp(-0.5 * (distance * distance) / (bandwidth * bandwidth));
    }

    template <typename T>
    void shift_point(const Point<T> &p, const vector<Point<T>> &points, double kernel_bandwidth, Point<T> &shifted_point, kernelFunc k) {
        shifted_point = p;
        if (points.size() == 1 || kernel_bandwidth == 0) {
            // nothing should be done
            return;
        }
        for (unsigned dim = 0; dim < shifted_point.pos.size(); dim++) {
            shifted_point.pos[dim] = 0;
        }
        double total_weight = 0;
        for (unsigned i = 0; i < points.size(); i++) {
            auto distance = euclidean_distance(p.pos, points[i].pos);
            auto weight = k(distance, kernel_bandwidth);
            for (unsigned j = 0; j < shifted_point.pos.size(); j++) {
                shifted_point.pos[j] += points[i].pos[j] * weight;
            }
            total_weight += weight;
        }

        if (total_weight == 0) {
            printf("shift_point: warning, total weight was 0\n");
            // would get NaNs if continuing
        } else {
            const double total_weight_inv = 1.0 / total_weight;
            for (unsigned i = 0; i < shifted_point.pos.size(); i++) {
                shifted_point.pos[i] *= total_weight_inv;
            }
        }
    }


    template <typename T>
    vector<Point<T>> meanshift(const vector<Point<T>> &points, double bandwidth, kernelFunc kernel, double EPSILON = 0.00001) {
        vector<Point<T>> shifted_points = points;
        const auto EPSILON_SQR = EPSILON * EPSILON;
        vector<bool> stop_moving(points.size(), false);
        double max_shift_distance;
        Point<T> point_new;
        do {
            max_shift_distance = 0;
            for (unsigned i = 0; i < points.size(); i++) {
                if (!stop_moving[i]) {
                    shift_point(shifted_points[i], points, bandwidth, point_new, kernel);
                    auto shift_distance_sqr = euclidean_distance_sqr(point_new.pos, shifted_points[i].pos);
                    if (shift_distance_sqr > max_shift_distance) {
                        max_shift_distance = shift_distance_sqr;
                    }
                    if (shift_distance_sqr <= EPSILON_SQR) {
                        stop_moving[i] = true;
                    }
                    shifted_points[i] = point_new;
                }
            }
            //printf("max_shift_distance: %f\n", sqrt(max_shift_distance));
        } while (max_shift_distance > EPSILON_SQR);

        return shifted_points;
    }

    template <typename T>
    vector<Cluster<T>> makeCluster(const vector<Point<T>> &points, const vector<Point<T>> &shifted_points) {
        vector<Cluster<T>>clusters;
        for (unsigned i = 0; i < shifted_points.size(); i++) {
            unsigned c = 0;
            // find cluster to add current point to
            while (c < clusters.size() && euclidean_distance(shifted_points[i].pos, clusters[c].getMode()) > CLUSTER_EPSILON) {
                c++;
            }
            if (c == clusters.size()) {
                // make new cluster
                clusters.push_back(ccCluster {shifted_points[i].pos});
            }
            clusters[c].addPoint(points[i], shifted_points[i]);
        }
        return clusters;
    }

    template <typename T>
    vector<Cluster<T>> cluster(const vector<Point<T>> &points, double kernel_bandwidth, kernelFunc k) {
        auto shifted_points = meanshift(points, kernel_bandwidth, k);
        auto cluster = makeCluster(points, shifted_points);
        return cluster;
    }

    // need to instantiate types that will be used
    template vector<Cluster<CC>> meanShift::cluster<CC>(const vector<Point<CC>>&, double, kernelFunc);
}

