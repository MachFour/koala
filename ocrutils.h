//
// Created by max on 9/23/18.
//

#ifndef REFERENCE_OCRUTILS_H
#define REFERENCE_OCRUTILS_H

#include <opencv2/core.hpp>

#ifdef REFERENCE_ANDROID
#include <baseapi.h>
#include <allheaders.h>
#else
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#endif

#include "wordBB.h"

int tesseractInit(tesseract::TessBaseAPI& baseAPI, const char * path, const char * dataPath);
std::string getCleanedText(tesseract::TessBaseAPI&, wordBB w);
std::string getCleanedText(tesseract::TessBaseAPI&, wordBB w, cv::Mat& tessImage);
cv::Mat matFromPix1(struct Pix * pix);

#endif //REFERENCE_OCRUTILS_H
