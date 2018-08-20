//
// Created by max on 8/19/18.
//


#include "randomColour.h"

cv::Scalar pseudoRandomColour(int a, int b, int c, int d, int minVal) {
    int modVal = 255-minVal;
    int multiplier = 31;
    int red = minVal + (multiplier * a * b * c % modVal);
    int green = minVal + (multiplier * b * c * d % modVal);
    int blue = minVal + (multiplier * d * a * b % modVal);
    return cv::Scalar(red, green, blue);
}

cv::Scalar pseudoRandomColour(int a, int b, int minVal) {
    return pseudoRandomColour(a, b, 19*a, 13*b, minVal);

}

cv::Scalar pseudoRandomColour(int a) {
    return pseudoRandomColour(1+a, 10 + a, 50+a, 200+a);

}
