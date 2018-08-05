//
// Created by max on 8/3/18.
//

#include <vector>
#include <algorithm>
#include "reference.h"
#include "ccomponent.h"

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
    return cv::getStructuringElement(shape, cv::Size(size, size) /*, cv::point anchor = cv::point(-1, -1) */);
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
        cv::Scalar colour = pseudoRandomColour(static_cast<int>(13* c.getSize()), ((int)c.getMode()[0] % 157));
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

void showRects(const Mat& image, const vector<cv::Rect>& overlappingCCRects) {
    Mat rectImg;
    cv::cvtColor(image, rectImg, CV_GRAY2BGR);
    for (const cv::Rect& r : overlappingCCRects) {
        cv::rectangle(rectImg, r, pseudoRandomColour(r.x, r.y, r.width, r.height), 5);
    }
    showImage(rectImg);
}

cv::Scalar pseudoRandomColour(int a, int b, int minVal) {
    return pseudoRandomColour(a, b, 3*a+5*b, a*b, minVal);

}
cv::Scalar pseudoRandomColour(int a, int b, int c, int d, int minVal) {
    int modVal = 255-minVal;
    int multiplier = 31;
    int red = minVal + (multiplier * a * b * c % modVal);
    int green = minVal + (multiplier * b * c * d % modVal);
    int blue = minVal + (multiplier * d * a * b % modVal);
    return cv::Scalar(red, green, blue);
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