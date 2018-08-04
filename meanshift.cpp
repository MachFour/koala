/*
 * Implementation for mean shift algorithm
 * Taken (and modified) from https://github.com/mattnedrich/MeanShift_cpp (licensed under MIT license)
 */

//#include <stdio.h>
#include <math.h>
#include "meanshift.h"

#define CLUSTER_EPSILON 0.5

namespace meanShift {
    double euclidean_distance_sqr(const position &a, const position &b) {
        double total = 0;
        for (unsigned i = 0; i < a.size(); i++) {
            const double temp = (a[i] - b[i]);
            total += temp * temp;
        }
        return total;
    }

    double euclidean_distance(const position &a, const position &b) {
        return sqrt(meanShift::euclidean_distance_sqr(a, b));
    }

    double gaussianKernel(double distance, double kernel_bandwidth) {
        double temp = exp(-0.5 * (distance * distance) / (kernel_bandwidth * kernel_bandwidth));
        return temp;
    }

    void shift_point(const Point &p, const std::vector<Point> &points, double kernel_bandwidth, Point &shifted_point, kernelFunc k) {
        shifted_point = p;
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

        const double total_weight_inv = 1.0 / total_weight;
        for (unsigned i = 0; i < shifted_point.pos.size(); i++) {
            shifted_point.pos[i] *= total_weight_inv;
        }
    }


    std::vector<Point> meanshift(const std::vector<Point> &points, double bandwidth, kernelFunc kernel, double EPSILON = 0.00001) {
        const auto EPSILON_SQR = EPSILON * EPSILON;
        std::vector<bool> stop_moving(points.size(), false);
        std::vector<Point> shifted_points = points;
        double max_shift_distance;
        Point point_new;
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

    ClusterList makeCluster(const std::vector<Point> &points, const std::vector<Point> &shifted_points) {
        ClusterList clusters;
        for (unsigned i = 0; i < shifted_points.size(); i++) {
            unsigned c = 0;
            // find cluster to add current point to
            while (c < clusters.size() && euclidean_distance(shifted_points[i].pos, clusters[c].mode) > CLUSTER_EPSILON) {
                c++;
            }
            if (c == clusters.size()) {
                // make new cluster
                Cluster clus;
                clus.mode = shifted_points[i].pos;
                clusters.push_back(clus);
            }
            clusters[c].original_points.push_back(points[i]);
            clusters[c].shifted_points.push_back(shifted_points[i]);
        }
        return clusters;
    }

    ClusterList cluster(const std::vector<Point> &points, double kernel_bandwidth, kernelFunc k) {
        auto shifted_points = meanshift(points, kernel_bandwidth, k);
        auto cluster = makeCluster(points, shifted_points);
        // sort by cluster size, in *descending* order
        return cluster;
    }

    void ClusterList::sortBySize(bool descending) {
        if (descending) {
            // sort ensures that the compare function resturns true on any two successive elements
            std::sort(begin(), end(), [](Cluster &c1, Cluster &c2) -> bool { return c1.size() > c2.size(); });
        } else {
            std::sort(begin(), end(), [](Cluster &c1, Cluster &c2) -> bool { return c1.size() < c2.size(); });
        }
    }
}
