//
// Created by max on 8/19/18.
//

#ifndef REFERENCE_RANDOMCOLOUR_H
#define REFERENCE_RANDOMCOLOUR_H

#include <opencv2/core.hpp>

cv::Scalar pseudoRandomColour(int);
cv::Scalar pseudoRandomColour(int, int, int minVal = 96);
cv::Scalar pseudoRandomColour(int, int, int, int, int minVal = 96);
#endif //REFERENCE_RANDOMCOLOUR_H
