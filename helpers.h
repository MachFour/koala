//
// Created by max on 9/11/18.
//
#ifndef REFERENCE_UTILS_H
#define REFERENCE_UTILS_H

#include "meanshift.h"
#include "wordBB.h"
#include "Interval.h"
#include "ccomponent.h"
#include "matutils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>





cv::Mat drawCentroidClusters(const cv::Mat&, const std::vector<ccCluster>&);
void showRowBounds(const cv::Mat&, const std::vector<ccCluster>&);

int findMedian(std::vector<int> numbers);

void drawCC(cv::Mat&, const CC&, cv::Scalar colour = cv::Scalar(255, 255, 255));
cv::Mat overlayWords(const cv::Mat &image, const std::vector<std::vector<wordBB>> &words, bool colourByRowCol=false);
cv::Mat overlayWords(const cv::Mat &image, const std::vector<wordBB> &allWordBBs, bool colourByRowCol=false);

/*
 * Use morphological opening / closing to equalise the brightness of an image,
 * by removing the 'small features' with the structuring element
 * Depending on the value of doDivide, one of two strategies is used:
 *  1. subtract the result of the morphological opening from the input image
 *  2. divide the input image by the result of the closing
 * The output image always has white on black text, like the input
 */
cv::Mat textEnhance(const cv::Mat& whiteOnBlack, const cv::Mat& structuringElement, bool doDivide);

std::string readFile(const std::string &filename);
std::string basename(std::string filename, bool removeExtension=false);



#endif //REFERENCE_UTILS_H
