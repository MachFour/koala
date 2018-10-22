//
// Created by max on 8/3/18.
//

#include "drawingutils.h"

#include "tableExtract.h"
#include "ccomponent.h"
#include "Interval.h"
#include "randomColour.h"
#include "matutils.h"

#include <opencv2/core.hpp>

#include <vector>

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

