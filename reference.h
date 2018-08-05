//
// Created by max on 8/2/18.
//

#ifndef REFERENCE_H
#define REFERENCE_H

#include <opencv2/opencv.hpp>
#include "meanshift.h"
#include "ccomponent.h"

using Mat = cv::Mat;
template <typename T>
using vector = std::vector<T>;

void showImage(const Mat& img);
int saveOrShowImage(const Mat&, const char *);
// image, labels, stats, centroids
void drawCC(Mat&, const CComponent&, cv::Scalar colour = cv::Scalar(255, 255, 255));

std::string type2str(int type);
Mat structuringElement(int size, cv::MorphShapes shape);

const static int BLUR_SIZE = 3;
const static int STDDEV_WIN = 32;
const static cv::Size STDDEV_WIN_SIZE = cv::Size(STDDEV_WIN, STDDEV_WIN);
const static double VARIANCE_THRESHOLD = 0.1;

void showCentroidClusters(const Mat&, const vector<ccCluster>&);
void showRowBounds(const Mat&, const vector<ccCluster>&);
void showRects(const Mat&, const vector<cv::Rect>&);

int findMedian(vector<int> numbers);

// pseudo random RGB colours, for different numbers of parameters
cv::Scalar pseudoRandomColour(int, int, int minVal = 96);
cv::Scalar pseudoRandomColour(int, int, int, int, int minVal = 96);


#endif //REFERENCE_H
