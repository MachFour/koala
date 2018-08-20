//
// Created by max on 8/3/18.
//

#include <leptonica/allheaders.h>

#include <vector>
#include <algorithm>

#include "reference.h"
#include "ccomponent.h"
#include "Interval.h"
#include "randomColour.h"

void drawCC(Mat& img, const CComponent& cc, cv::Scalar colour) {
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
    cv::rectangle(img, cv::Point(cc.left, cc.top), cv::Point(cc.left+cc.width, cc.top+cc.height), colour, /*thickness=*/5);


}

cv::Rect findBoundingRect(const vector<Interval>& intervals, const vector<CComponent>& allCCs, int maxHeight, int maxWidth) {
    int minTop = maxHeight; // >= anything inside image
    int minLeft = maxWidth; // >= anything inside image
    int maxBottom = 0; // <= smaller than anything inside image
    int maxRight = 0; // <= smaller than anything inside image
    for (const Interval& iv : intervals) {
        // TODO avoid indexing back into global list
        int label = iv.getLabel();
        int top = allCCs[label].top;
        int height = allCCs[label].height;
        int width = allCCs[label].width;
        int left = allCCs[label].left;
        minTop = cv::min(top, minTop);
        minLeft = cv::min(left, minLeft);
        maxBottom = cv::max(top+height, maxBottom);
        maxRight = cv::max(left+width, maxRight);
    }
    // left(x), top(y), width, height
    int rectHeight = maxBottom - minTop;
    int rectWidth = maxRight - minLeft;

    // simple filtering
    return cv::Rect(minLeft, minTop, rectWidth, rectHeight);
}

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

Mat structuringElement(int size, cv::MorphShapes shape) {
    // point (-1, -1) represents centred structuring element
    return structuringElement(size, size, shape);
}
Mat structuringElement(int width, int height, cv::MorphShapes shape) {
    // point (-1, -1) represents centred structuring element
    return cv::getStructuringElement(shape, cv::Size(width, height) /*, cv::point anchor = cv::point(-1, -1) */);
}

void showImage(const Mat& img) {
    saveOrShowImage(img, "show");
}

int findMedian(vector<int> numbers) {
    auto n = numbers.size();
    if (n <= 0) {
        return 0;
    }
    std::sort(numbers.begin(), numbers.end());

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

void showCentroidClusters(const Mat& image, const vector<ccCluster>& clustersByCentroid) {
    // now draw on the clustered CCs
    Mat clusteredCCs;
    cv::cvtColor(image, clusteredCCs, CV_GRAY2BGR);
    for (const ccCluster &c : clustersByCentroid) {
        // pick a pseudorandom nice colour
        cv::Scalar colour = pseudoRandomColour(13* c.getSize(), ((int)c.getMode()[0]) % 157);
        for (CComponent cc : c.getData()) {
            drawCC(clusteredCCs, cc, colour);
        }
    }
    showImage(clusteredCCs);
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
        int clusterSize = static_cast<int>(c.getSize());
        maxSize = cv::max(clusterSize, maxSize);
        for (const CComponent& cc : c.getData()) {
            minY = cv::min(minY, cc.top);
            maxY = cv::max(maxY, cc.top + cc.height);
        }
        // rect parameters: x, y, width, height
        // we'll use width for getSize, height for height
        rowBounds.push_back(cv::Rect(0, minY, clusterSize, maxY - minY));
    }

    // now draw on rows
    Mat rowsImg;
    cv::cvtColor(image, rowsImg, CV_GRAY2BGR);
    for (cv::Rect& r : rowBounds) {
        // choose colour
        // make width proportional to relative getSize of cluster
        int width = static_cast<int>(round(((double)r.width/maxSize)*rowsImg.cols));
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


// returns 8-bit single channel Mat from corresponding binary pix image
cv::Mat matFromPix1(PIX * pix) {
    Mat mat(pix->h, pix->w, CV_8UC1);
    // loop inspired by function ImageThresholder::SetImage from src/ccmain/thresholder.cpp of Tesseract
    // located at: https://github.com/tesseract-ocr/tesseract/tree/master/src/ccmain/thresholder.cpp

    l_uint32* data = pixGetData(pix);
    int wpl = pixGetWpl(pix);
    for (unsigned int y = 0; y < pix->h; ++y, data += wpl) {
        for (unsigned int x = 0; x < pix->w; ++x) {
            mat.at<unsigned char>(y, x) = (unsigned char) (GET_DATA_BIT(data, x) ? 255 : 0);
        }
    }
    return mat;
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

cv::Ptr<Plot> makePlot(const Mat &data, const Mat *resize, cv::Scalar colour, int thickness) {
    cv::Ptr<Plot> plot = Plot::create(data);
    plot->setNeedPlotLine(true);
    plot->setShowGrid(false);
    plot->setPlotLineWidth(thickness);
    plot->setPlotLineColor(colour);
    if (resize != nullptr) {
        plot->setPlotSize(resize->cols, resize->rows);
    }
    plot->setInvertOrientation(true);

    return plot;
}

int saveOrShowImage(const Mat& img, const char * outFile) {
    bool isForDisplay = strcmp(outFile, "show") == 0;
    if (isForDisplay) {
        cv::namedWindow("output", cv::WINDOW_NORMAL);
        cv::imshow("output", img);
        cv::resizeWindow("output", 1024, 768);
        cv::waitKey(0);
        return 0;
    } // else {

    bool result = true;
    try {
        result &= cv::imwrite(outFile, img);
    } catch (const cv::Exception& ex) {
        fprintf(stderr, "exception writing images: %s\n", ex.what());
    }

    if (!result) {
        fprintf(stderr, "Error saving images\n");
        return 1;
    } else {
        return 0;
    }
}
