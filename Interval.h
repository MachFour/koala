//
// Created by max on 8/2/18.
//

#ifndef REFERENCE_INTERVAL_H
#define REFERENCE_INTERVAL_H

#include <vector>
#include <cmath>

class Interval {
public:
    // how to apply a length expansion when determining whether two line segments are close
    enum ExpandType {
          MIN // use the shorter interval's length to expand
        , MAX // use the longer interval's length to expand
        , AVG // use the average of the two intervals' lengths to expand
    };

    Interval(int label, double point1, double point2) :
        label(label), midpoint(halfway(point1, point2)), halflength(halfDistance(point1, point2)) {};

    double left() const { return midpoint - halflength; }
    double right() const { return midpoint + halflength; }
    int getLabel() const { return label; }



    // intervals vector needs to be copied because it's sorted
    static void groupCloseIntervals(std::vector<Interval>, std::vector<std::vector<Interval>>&, double, ExpandType method = AVG);
    static bool areClose(const Interval&, const Interval&, double expand = 1.0, ExpandType method = AVG);
    static bool closeToAny(const Interval&, const std::vector<Interval>& others, double expand = 1.0, ExpandType method = AVG);

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

#endif //REFERENCE_INTERVAL_H
