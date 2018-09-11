//
// Created by max on 9/11/18.
//
#ifndef REFERENCE_UTILS_H
#define REFERENCE_UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#ifndef REFERENCE_ANDROID
#include <leptonica/allheaders.h>
#else
#include <allheaders.h>
#endif

#include <vector>
#include "meanshift.h"
#include "wordBB.h"
#include "Interval.h"

#include "ccomponent.h"
#include "wordBB.h"
#include "Interval.h"

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

cv::Mat matFromPix1(struct Pix * pix);

Mat derivative(const Mat& src);

int saveOrShowImage(const Mat&, const char *);

// pseudo random RGB colours, for different numbers of parameters

#endif //REFERENCE_UTILS_H
