//
// Created by max on 9/11/18.
//
#ifndef REFERENCE_UTILS_H
#define REFERENCE_UTILS_H

#include "meanshift.h"
#include "wordBB.h"
#include "Interval.h"
#include "ccomponent.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

using Mat = cv::Mat;
template <typename T>
using vector = std::vector<T>;


// image, labels, stats, centroids
void drawCC(Mat&, const CComponent&, cv::Scalar colour = cv::Scalar(255, 255, 255));

cv::Rect findBoundingRect(const vector<Interval>&, const vector<CComponent>&, int maxHeight, int maxWidth);

std::string type2str(int type);

Mat structuringElement(int size, cv::MorphShapes shape);

Mat structuringElement(int width, int height, cv::MorphShapes shape);

void showImage(const Mat& img);

int findMedian(vector<int> numbers);

void showCentroidClusters(const Mat&, const vector<ccCluster>&);

void showRowBounds(const Mat&, const vector<ccCluster>&);

Mat overlayWords(const Mat &image, const vector<vector<wordBB>> &words, bool colourByRowCol=false);

Mat overlayWords(const Mat &image, const vector<wordBB> &allWordBBs, bool colourByRowCol=false);

std::string readFile(const std::string &filename);
std::string basename(std::string filename, bool removeExtension=false);

Mat derivative(const Mat& src);

int saveImage(const Mat &, const char *);

#endif //REFERENCE_UTILS_H
