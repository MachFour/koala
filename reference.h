//
// Created by max on 8/2/18.
//

#ifndef REFERENCE_H
#define REFERENCE_H

#include "table.h"
#include "wordBB.h"

#include <opencv2/core.hpp>

#ifdef REFERENCE_ANDROID
#include <baseapi.h>
#else
#include <tesseract/baseapi.h>
#endif


const static int BLUR_SIZE = 3;
const static int STDDEV_WIN = 32;
const static cv::Size STDDEV_WIN_SIZE = cv::Size(STDDEV_WIN, STDDEV_WIN);
const static double VARIANCE_THRESHOLD = 0.1;

/*
 * what we're all here for
 */
Table tableExtract(const cv::Mat &img, tesseract::TessBaseAPI& tessApi, cv::Mat * wordBBImg = nullptr, bool batchMode = true);

#endif //REFERENCE_H
