//
// Created by max on 10/9/18.
//

#include "matutils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <stdexcept>
#include <algorithm>

using std::string;
using cv::Mat;

std::string type2str(int type) {
    std::string r;

    int depth = type & CV_MAT_DEPTH_MASK;
    int chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

Mat eightBitToFloat(const Mat& input, bool rescale, bool doublePrecision) {
    if (input.depth() != CV_8U) {
        throw std::invalid_argument("Mat is not CV_8U");
    }
    Mat ret;
    input.convertTo(ret, doublePrecision ? CV_64F : CV_32F, rescale ? 1.0/255 : 1);
    return ret;
}

Mat floatToEightBit(const Mat& input, bool rescale) {
    if (input.depth() != CV_32F && input.depth() != CV_64F) {
        throw std::invalid_argument("Mat is not CV_32F or CV_64F");
    }
    Mat ret;
    input.convertTo(ret, CV_8U, rescale ? 255 : 1);
    return ret;
}


Mat invert(const Mat& input) {
    switch (input.depth()) {
        case CV_8U:
            return 255 - input;
        case CV_64F:
            return 1.0 - input;
        case CV_32F:
            return 1.0f - input;
        default:
            throw std::invalid_argument("Unsupported Mat type");
    }
}

/*
 * Maximum value of a particular matrix type
 */
int maxVal(int matDepth) {
    switch (matDepth) {
        case CV_8U:
            return 255;
        case CV_32F:
            /* fall through */
        case CV_64F:
            return 1;
        case CV_32S:
            return std::numeric_limits<int>::max();
        default:
            throw std::invalid_argument("Unrecognised Mat depth");
    }

}

int findMedian(std::vector<int> numbers) {
    auto n = numbers.size();
    if (n <= 0) {
        return 0;
    }
    sort(numbers.begin(), numbers.end());

    if (n % 2 == 1) {
        // n == 5 -> return numbers[2]
        return numbers[(n - 1) / 2];
    } else {
        // n == 6 -> return round(numbers[2] + numbers[3])/2
        int smallerMedian = numbers[n/2 - 1];
        int largerMedian = numbers[n/2];
        return static_cast<int>(round((smallerMedian + largerMedian)/2.0));
    }
}

void showImage(const Mat& img, const std::string& title) {
#ifndef REFERENCE_ANDROID
    constexpr auto winname = "CV_IMAGE";
    namedWindow(winname, cv::WINDOW_NORMAL);
    cv::imshow(winname, img);
    cv::resizeWindow(winname, 1024, 768);
    if (!title.empty()) {
        cv::setWindowTitle(winname, title);
    }
    cv::waitKey(0);
#endif
}

// element-wise derivative/difference along first dimension
// matrix must be CV_64F
Mat derivative(const Mat& src) {
    Mat deriv(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int x = 0; x < src.cols; ++x) {
        for (int y = 1; y < src.rows; ++y) {
            deriv.at<double>(y, x) = src.at<double>(y, x) - src.at<double>(y-1, x);
        }
    }
    return deriv;
}



int saveImage(const Mat &img, const char *outFile) {
    bool result = false;
#ifndef REFERENCE_ANDROID
    try {
        result |= cv::imwrite(outFile, img);
        if (!result) {
            fprintf(stderr, "Error saving images\n");
        }
    } catch (const cv::Exception& ex) {
        fprintf(stderr, "exception writing images: %s\n", ex.what());
    }
#endif
    // result = true -> return 0 for no error
    return !result;
}
