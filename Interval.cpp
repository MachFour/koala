//
// Created by max on 8/2/18.
//

#include "Interval.h"

#include <vector>
#include <cmath>



int Interval::label() const {
    return _label;
}

void Interval::setLabel(int label) {
    _label = label;
}

double Interval::left() const {
    return midpoint - halflength;
}

double Interval::right() const {
    return midpoint + halflength;
}

bool Interval::areClose(const Interval &i1, const Interval &i2, double expand, ExpandType method) {
    double dist = distance(i1.midpoint, i2.midpoint);
    double longerHL = std::fmax(i1.halflength, i2.halflength);
    double shorterHL = std::fmin(i1.halflength, i2.halflength);
    switch (method) {
        case MIN:
            return dist < longerHL + shorterHL*expand*2;
        case MAX:
            return dist < longerHL*expand*2 + shorterHL;
        case AVG:
        default:
            return dist < longerHL*expand + shorterHL*expand;
    }
}

bool Interval::closeToAny(const Interval &one, const std::vector<Interval> &others, double expand, ExpandType method) {
    for (const Interval& other : others) {
        if (areClose(one, other, expand, method)) {
            return true;
        }
    }
    return false;
}

double Interval::halfway(double a, double b) {
    return (a + b)/2.0;
}
double Interval::distance(double a, double b) {
    return std::fabs(a - b);
}
double Interval::halfDistance(double a, double b) {
    return distance(a, b)/2.0;
}
