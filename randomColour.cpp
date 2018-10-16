//
// Created by max on 8/19/18.
//

#include "randomColour.h"

cv::Scalar pseudoRandomColour(int a, int b, int c, int d, int minVal) {
    int modVal = 255-minVal;
    int multiplier = 31;
    auto red   = static_cast<double>(minVal + (multiplier * a * b * c % modVal));
    auto green = static_cast<double>(minVal + (multiplier * b * c * d % modVal));
    auto blue  = static_cast<double>(minVal + (multiplier * d * a * b % modVal));
    return {red, green, blue};
}

cv::Scalar pseudoRandomColour(int a, int b, int minVal) {
    return pseudoRandomColour(a, b, 19*a, 13*b, minVal);
}

cv::Scalar pseudoRandomColour(int a) {
    return pseudoRandomColour(1 + a, 11 + 2*a, 51 + 3*a, 479 + 4*a);
}
