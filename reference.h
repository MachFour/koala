//
// Created by max on 8/2/18.
//

#ifndef REFERENCE_H
#define REFERENCE_H

#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <leptonica/allheaders.h>
#include "meanshift.h"
#include "ccomponent.h"
#include "wordBB.h"
#include "Interval.h"

using Mat = cv::Mat;
using Plot = cv::plot::Plot2d;
template <typename T>
using vector = std::vector<T>;

void showImage(const Mat& img);
int saveOrShowImage(const Mat&, const char *);
// image, labels, stats, centroids
void drawCC(Mat&, const CComponent&, cv::Scalar colour = cv::Scalar(255, 255, 255));

std::string type2str(int type);
Mat structuringElement(int size, cv::MorphShapes shape);
Mat structuringElement(int width, int height, cv::MorphShapes shape);


const static int BLUR_SIZE = 3;
const static int STDDEV_WIN = 32;
const static cv::Size STDDEV_WIN_SIZE = cv::Size(STDDEV_WIN, STDDEV_WIN);
const static double VARIANCE_THRESHOLD = 0.1;

void showCentroidClusters(const Mat&, const vector<ccCluster>&);
void showRowBounds(const Mat&, const vector<ccCluster>&);
Mat overlayWords(const Mat &image, const vector<vector<wordBB>> &words, bool colourByRowCol=false);
Mat overlayWords(const Mat &image, const vector<wordBB> &allWordBBs, bool colourByRowCol=false);

int findMedian(vector<int> numbers);
cv::Rect findBoundingRect(const vector<Interval>&, const vector<CComponent>&, int maxHeight, int maxWidth);
cv::Mat matFromPix1(PIX * pix);
cv::Ptr<Plot>
makePlot(const Mat &data, const Mat *resize = nullptr, cv::Scalar colour = cv::Scalar(0, 255, 255), int thickness = 7);

Mat derivative(const Mat& src);

// pseudo random RGB colours, for different numbers of parameters


#endif //REFERENCE_H
