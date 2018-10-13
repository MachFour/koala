//
// Created by max on 8/2/18.
//

#ifndef REFERENCE_INTERVAL_H
#define REFERENCE_INTERVAL_H

#include <vector>
#include <functional>

class Interval {
public:
    // how to apply a length expansion when determining whether two line segments are close
    enum ExpandType {
          MIN // use the shorter interval's length to expand
        , MAX // use the longer interval's length to expand
        , AVG // use the average of the two intervals' lengths to expand
    };

    Interval(double point1, double point2) : Interval(0, point1, point2) {}

    Interval(int label, double point1, double point2) :
        _label(label), midpoint(halfway(point1, point2)), halflength(halfDistance(point1, point2)) {}

    int label() const;
    void setLabel(int label);
    double left() const;
    double right() const;

    // intervals vector needs to be copied because it's sorted


    template <typename T>
    static std::vector<std::vector<T>> groupClose(const std::vector<T>& things, double expansion, ExpandType method, std::function<Interval(T)> makeInterval);
    static bool areClose(const Interval&, const Interval&, double expand = 1.0, ExpandType method = AVG);
    static bool closeToAny(const Interval&, const std::vector<Interval>& others, double expand = 1.0, ExpandType method = AVG);

private:
    int _label;
    double midpoint;
    double halflength;

    static double halfway(double a, double b);
    static double distance(double a, double b);
    static double halfDistance(double a, double b);
};

#endif //REFERENCE_INTERVAL_H
