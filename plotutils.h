//
// Created by max on 9/11/18.
//

#ifndef REFERENCE_PLOTUTILS_H
#define REFERENCE_PLOTUTILS_H

#ifndef REFERENCE_ANDROID

#include <opencv2/plot.hpp>


cv::Ptr<cv::plot::Plot2d>
makePlot(const cv::Mat &data, const cv::Mat *resize = nullptr, cv::Scalar colour = cv::Scalar(0, 255, 255), int thickness = 7);

cv::Ptr<cv::plot::Plot2d>
makePlot(const cv::Mat &data, cv::Scalar colour = cv::Scalar(0, 255, 255), int thickness = 7, int resizeW=-1, int resizeH=-1);

#endif // REFERENCE_ANDROID

#endif //REFERENCE_PLOTUTILS_H
