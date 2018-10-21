//
// Created by max on 8/3/18.
//

#include "tableExtract.h"
#include "ccomponent.h"
#include "Interval.h"
#include "randomColour.h"
#include "helpers.h"

#include <opencv2/core.hpp>

#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <ios>

using std::vector;
using cv::Mat;


void drawCC(Mat& img, const CC& cc, cv::Scalar colour) {
    //int area = stats.at<int>(which, cv::CC_STAT_AREA);
    //int centroidX = static_cast<int>(centroids.at<double>(which, 0));
    //int centroidY = static_cast<int>(centroids.at<double>(which, 1));
    //std::cout << "data " << i << std::endl;
    //std::cout << "area: " << area << std::endl;
    //std::cout << "height: " << height << std::endl;
    //std::cout << "width: " << width << std::endl;
    //std::cout << "left: " << left << std::endl;
    //std::cout << "top: " << top << std::endl;
    //std::cout << "centroid: (" << centroidX << ", " << centroidY << ")" << std::endl;

    /* to extract the actual CC pixels
    Mat ithComponent;
    cv::compare(labels, which, ithComponent, cv::CMP_EQ);
     */
    //cv::drawMarker(img, cv::Point(centroidX, centroidY), /*colour=*/255, cv::MARKER_TILTED_CROSS, /*getSize=*/20, /*thickness=*/3);
    //int thickness = static_cast<int>(1+log(area));
    cv::rectangle(img, cv::Point(cc.left(), cc.top()), cv::Point(cc.right(), cc.bottom()), colour, /*thickness=*/5);


}

int findMedian(vector<int> numbers) {
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

Mat drawCentroidClusters(const Mat& image, const vector<ccCluster>& clustersByCentroid) {
    // now draw on the clustered CCs
    Mat clusteredCCs;
    cv::cvtColor(image, clusteredCCs, CV_GRAY2BGR);
    for (const ccCluster &c : clustersByCentroid) {
        // pick a pseudorandom nice colour
        cv::Scalar colour = pseudoRandomColour((int)(13 * c.getSize()), ((int)c.getMode()[0]) % 157);
        for (const CC& cc : c.getData()) {
            drawCC(clusteredCCs, cc, colour);
        }
    }
    return clusteredCCs;
}

// input must be either CV_8U, CV_32F or CV_64F
cv::Mat textEnhance(const Mat& whiteOnBlack, const Mat& structuringElement, bool doDivide, std::vector<progressImg>& progressImages) {
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

void showRowBounds(const Mat& image, const vector<ccCluster>& clustersByCentroid) {
    /*
     * for each row, find top and bottom bounds
     */
    auto numRows = clustersByCentroid.size();
    vector<cv::Rect> rowBounds(numRows);
    int maxSize = 0;
    for (const ccCluster &c : clustersByCentroid) {
        int minY = image.rows;
        int maxY = 0;
        int clusterSize = (int) c.getSize();
        maxSize = std::max(clusterSize, maxSize);
        for (const CC& cc : c.getData()) {
            minY = std::min(minY, cc.top());
            maxY = std::max(maxY, cc.bottom());
        }
        // rect parameters: x, y, width, height
        // we'll use width for getSize, height for height
        rowBounds.emplace_back(cv::Rect(0, minY, clusterSize, maxY - minY));
    }

    // now draw on rows
    Mat rowsImg;
    cv::cvtColor(image, rowsImg, CV_GRAY2BGR);
    for (cv::Rect& r : rowBounds) {
        // choose colour
        // make width proportional to relative getSize of cluster
        int width = static_cast<int>(round(((double)r.width / maxSize) * rowsImg.cols));
        int top = r.y;
        int bottom = r.y + r.height;
        cv::rectangle(rowsImg, cv::Point(0, top), cv::Point(width, bottom), pseudoRandomColour(width, top), 5);
    }
    showImage(rowsImg);
}

Mat overlayWords(const Mat &image, const vector<vector<wordBB>> &rows, bool colourByRowCol) {
    Mat rectImg;
    cv::cvtColor(image, rectImg, CV_GRAY2BGR);
    for (const auto &row : rows) {
        for (const wordBB &w : row) {
            cv::rectangle(rectImg, w.asRect(), w.getColour(colourByRowCol), 5);
        }
    }
    return rectImg;
}

Mat overlayWords(const Mat &image, const vector<wordBB> &allWordBBs, bool colourByCol) {
    Mat rectImg;
    cv::cvtColor(image, rectImg, CV_GRAY2BGR);
    for (const wordBB &w : allWordBBs) {
        cv::rectangle(rectImg, w.asRect(), w.getColour(colourByCol), 5);
    }
    return rectImg;
}


// https://codereview.stackexchange.com/questions/22901/reading-all-bytes-from-a-file
// https://en.cppreference.com/w/cpp/io/basic_istream/read
std::string readFile(const std::string &filename) {
    // initially seek to end of file to get its position
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        fprintf(stderr, "Could not open file with name %s\n", filename.c_str());
        return "";
    }
    // XXX this may not always be an accurate indicator of the file size!
    std::fstream::pos_type end = ifs.tellg();
    if (end == std::fstream::pos_type(-1)) {
        // error
        fprintf(stderr, "readFile(): error end-of-file position: %s\n", filename.c_str());
        return "";
    }

    auto size = static_cast<size_t>(end);
#ifdef REFERENCE_ANDROID
#warning "tellg() may not work as a size indication on Android"
#endif

    std::string fileString(size, '\0');
    ifs.seekg(0);
    ifs.read(&fileString[0], size);
    ifs.close();

    return fileString;
}

std::string basename(std::string filename, bool removeExtension) {
    // check for and remove trailing slash(es)
    while (filename.back() == '/') {
        filename.pop_back();
    }
    size_t lastSlash = filename.rfind('/');
    if (lastSlash != std::string::npos) {
        // guaranteed that slash was not the last character, so there's at least one more character after lastSlash
        filename.erase(0, lastSlash+1);
    }
    size_t lastDot = filename.rfind('.');
    if (removeExtension && lastDot != std::string::npos) {
        filename.erase(lastDot, filename.size() - lastDot);
    }
    filename.shrink_to_fit();
    return filename;
}


