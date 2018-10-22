//
// Created by max on 9/11/18.
//
#ifndef REFERENCE_UTILS_H
#define REFERENCE_UTILS_H

#include "meanshift.h"
#include "wordBB.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>




cv::Mat drawCentroidClusters(const cv::Mat&, const std::vector<ccCluster>&);
void showRowBounds(const cv::Mat&, const std::vector<ccCluster>&);


void drawCC(cv::Mat&, const CC&, cv::Scalar colour = cv::Scalar(255, 255, 255));
cv::Mat overlayWords(const cv::Mat &image, const std::vector<std::vector<wordBB>> &words, bool colourByRowCol=false);
cv::Mat overlayWords(const cv::Mat &image, const std::vector<wordBB> &allWordBBs, bool colourByRowCol=false);



#endif //REFERENCE_UTILS_H
