//
// Created by max on 10/22/18.
//

#include "preprocessing.h"

#include "matutils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

using std::vector;
using cv::Mat;

static Mat structuringElement(int width, int height, cv::MorphShapes shape) {
    // point (-1, -1) represents centred structuring element
    return getStructuringElement(shape, cv::Size(width, height) /*, cv::point anchor = cv::point(-1, -1) */);
}

static Mat structuringElement(int size, cv::MorphShapes shape) {
    // point (-1, -1) represents centred structuring element
    return structuringElement(size, size, shape);
}

/*
 * Use morphological opening / closing to equalise the brightness of an image,
 * by removing the 'small features' with the structuring element
 * Depending on the value of doDivide, one of two strategies is used:
 *  1. subtract the result of the morphological opening from the input image
 *  2. divide the input image by the result of the closing
 * The output image always has white on black text, like the input
 */

// input must be either CV_8U, CV_32F or CV_64F
Mat textEnhance(const Mat& whiteOnBlack, const Mat& structuringElement, bool doDivide, vector<progressImg>& progressImages) {
    Mat background;
    Mat unnormalised;
    if (doDivide) {
        /* strategy from https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
         * to have uniform brightness, divide image by result of closure
         * needs a black on white image
         */
        Mat blackOnWhite = invert(whiteOnBlack);
        Mat lightBackground;
        cv::morphologyEx(blackOnWhite, lightBackground, cv::MorphTypes::MORPH_CLOSE, structuringElement);

        // need to use floating point Mat for division
        Mat tmp;
        if (blackOnWhite.depth() == CV_8U) {
            cv::divide(eightBitToFloat(blackOnWhite), eightBitToFloat(lightBackground), tmp);
        } else {
            cv::divide(blackOnWhite, lightBackground, tmp);
        }
        background = lightBackground;
        unnormalised = invert(tmp);
    } else {
        Mat darkBackground;
        cv::morphologyEx(whiteOnBlack, darkBackground, cv::MorphTypes::MORPH_OPEN, structuringElement);
        cv::subtract(whiteOnBlack, darkBackground, unnormalised);
        background = darkBackground;
    }
    // output bitness is same as input
    auto outDepth = whiteOnBlack.depth();

    Mat normalised;
    cv::normalize(unnormalised, normalised, 0, maxVal(outDepth), cv::NORM_MINMAX, outDepth);
    progressImages.emplace_back(progressImg{background, "background"});
    progressImages.emplace_back(progressImg{unnormalised, "enhanced-unnormalised"});
    progressImages.emplace_back(progressImg{normalised, "enhanced-normalised"});

    return normalised;
}

// do dumb threshold to figure out whether it's white on black or black on white text, and invert if necessary
// assumes that there is actually a clear majority of one over the other (i.e many more background pixels than foreground)

static bool isWhiteTextOnBlack(const Mat& m, std::vector<progressImg>& progressImages) {
    if (m.depth() != CV_8U) {
        throw std::invalid_argument("matrix must be CV_8U");
    }
    Mat equalized;
    cv::equalizeHist(m, equalized);
    progressImages.emplace_back(progressImg{equalized, "equalized"});

    Mat blurred;
    cv::medianBlur(equalized, blurred, 11);
    progressImages.emplace_back(progressImg{blurred, "blurred"});


    Mat thresholded;
    //cv::threshold(thresholded, thresholded, 0, 255, cv::THRESH_OTSU);
    cv::adaptiveThreshold(blurred, thresholded, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 399, 0);

    progressImages.emplace_back(progressImg{thresholded, "thresholded"});
    // if nonzero pixels make up less than half the area of the image, it's probably white on black text
    return cv::countNonZero(thresholded) < m.rows * m.cols / 2;
}

Mat preprocess(const cv::Mat& grey8, vector<progressImg>& progressImages) {

    // do dumb threshold to figure out whether it's white on black or black on white text, and invert if necessary
    Mat whiteOnBlack = isWhiteTextOnBlack(grey8, progressImages) ? grey8 : invert(grey8);
    progressImages.emplace_back(whiteOnBlack, "whiteOnBlack");

    const auto openingKsize = std::max(whiteOnBlack.rows, whiteOnBlack.cols)/30;
    const Mat sElement = structuringElement(openingKsize, cv::MORPH_RECT);
    Mat textEnhanced = textEnhance(whiteOnBlack, sElement, false, progressImages);


    Mat vLines;
    Mat hLines;
    // detect large horizontal and vertical lines
    cv::morphologyEx(textEnhanced, hLines, cv::MorphTypes::MORPH_OPEN, structuringElement(250, 5, cv::MORPH_RECT));
    cv::morphologyEx(textEnhanced, vLines, cv::MorphTypes::MORPH_OPEN, structuringElement(5, 250, cv::MORPH_RECT));

    //Mat opened;
    //cv::morphologyEx(linesRemoved, opened, cv::MorphTypes::MORPH_OPEN, structuringElement(12, 12, cv::MORPH_ELLIPSE));

    // (gaussian) blur -> matches vision character
    // before doing the morphological operation - makes intensities more uniform in th
    // don't use the binarised (or blurred) image for OCR (don't throw away)
    // histogram of gradients

    //clean it up a bit?
    // shapes: MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    //cv::morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_ERODE, structuringElement(2, cv::MORPH_ELLIPSE));
    //cv::morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_OPEN, structuringElement(1, 7, cv::MORPH_RECT));

    //int C = 0; // constant subtracted from calculated threshold value to obtain T(x, y)
    //cv::adaptiveThreshold(open, binarised, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 399, C);
    //cv::morphologyEx(binarised, binarised, cv::MorphTypes::MORPH_OPEN, structuringElement(2, cv::MORPH_ELLIPSE));
    //cv::threshold(hLines, hLinesBin, 0, 255, cv::THRESH_TOZERO | cv::THRESH_OTSU);
    //cv::threshold(vLines, vLinesBin, 0, 255, cv::THRESH_TOZERO | cv::THRESH_OTSU);

    // remove lines
    Mat preprocessed;
    cv::subtract(textEnhanced, hLines, preprocessed, cv::noArray(), CV_8U);
    cv::subtract(preprocessed, vLines, preprocessed, cv::noArray(), CV_8U);
    {
        progressImages.emplace_back(progressImg{grey8, "image"});
        progressImages.emplace_back(progressImg{textEnhanced, "textEnhanced"});
        progressImages.emplace_back(progressImg{hLines, "horizontal lines"});
        progressImages.emplace_back(progressImg{vLines, "vertical lines"});
        progressImages.emplace_back(progressImg{preprocessed, "preprocessed"});
    }

    return preprocessed;

}
