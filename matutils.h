//
// Created by max on 10/9/18.
//

#ifndef REFERENCE_MATUTILS_H
#define REFERENCE_MATUTILS_H


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

cv::Mat floatToEightBit(const cv::Mat& input, bool rescale=true);
cv::Mat eightBitToFloat(const cv::Mat& input, bool rescale=true, bool doublePrecision=true);
cv::Mat invert(const cv::Mat& input);

int maxVal(int matDepth);

cv::Mat derivative(const cv::Mat& src);
cv::Mat structuringElement(int size, cv::MorphShapes shape);
cv::Mat structuringElement(int width, int height, cv::MorphShapes shape);


// pair of Mat and title
using progressImg = std::pair<cv::Mat, std::string>;
bool isWhiteTextOnBlack(const cv::Mat& m, std::vector<progressImg>& progressImages);


std::string type2str(int type);
void showImage(const cv::Mat& img, const std::string& title="");
int saveImage(const cv::Mat &, const char *);

#endif //REFERENCE_MATUTILS_H
