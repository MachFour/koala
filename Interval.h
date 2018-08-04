//
// Created by max on 8/2/18.
//

#ifndef REFERENCE_INTERVAL_H
#define REFERENCE_INTERVAL_H

#include <vector>
#include <cmath>

template <typename T>
using vector = std::vector<T>;

class Interval {
public:
    // how to decide whether two line segments 'overlap' when using a fudge factor
    enum OverlapType {
          MIN // measure closeness relative to shorter of the two intervals
        , MAX // measure closeness relative to larger of the two intervals
        , AVG // measure closeness relative to average length of the two intervals
    };

    Interval(int label, double point1, double point2) :
        label(label), midpoint(halfway(point1, point2)), halflength(halfDistance(point1, point2)) {};

    double left() const { return midpoint - halflength; }
    double right() const { return midpoint + halflength; }
    int getLabel() const { return label; }

    // intervals vector needs to be copied because it's sorted
    static void groupCloseIntervals(vector<Interval> intervals, vector<vector<Interval>> &partitions, double expansion);
    static bool areClose(const Interval &i1, const Interval &i2, double expand = 1.0, OverlapType method = AVG);

private:
    int label;
    double midpoint;
    double halflength;

    static double halfway(double a, double b) {
        return (a + b)/2.0;
    }
    static double distance(double a, double b) {
        return fabs(a - b);
    }
    static double halfDistance(double a, double b) {
        return distance(a, b)/2.0;
    }

};

#endif //REFERENCE_INdoubleERVAL_H
