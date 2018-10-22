//
// Created by max on 10/22/18.
//

#ifndef KOALA_PREPROCESSING_H
#define KOALA_PREPROCESSING_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

// pair of Mat and title
using progressImg = std::pair<cv::Mat, std::string>;



/*
 * The overall preprocessing function. Helper functions are inside preprocessing.cpp
 */
cv::Mat preprocess(const cv::Mat&, std::vector<progressImg>&);

#endif //KOALA_PREPROCESSING_H
