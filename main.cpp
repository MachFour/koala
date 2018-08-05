#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "reference.h"
#include "meanshift.h"
#include "Interval.h"
#include "ccomponent.h"
//#include <opencv2/ximgproc.hpp>

int main(int argc, char ** argv) {
    if (argc != 3) {
        printf("Usage: %s <input.img> <output.img|show>\n", argv[0]);
        return -1;
    }

    char * const inFile = argv[1];
    Mat image = imread(inFile, cv::IMREAD_GRAYSCALE);
    if (!image.data) {
        fprintf(stderr, "Could not read input image!\n");
        return 1;
    }

    Mat preprocessed;
    image.convertTo(preprocessed, CV_8UC1);

    // do dumb threshold to figure out whether it's white on black or black on white text, and invert if necessary
    Mat dumbThreshold;
    cv::equalizeHist(preprocessed, dumbThreshold);
    //cv::threshold(dumbThreshold, dumbThreshold, 0, 255, cv::THRESH_OTSU);
    cv::adaptiveThreshold(dumbThreshold, dumbThreshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 399, 0);
    if (cv::countNonZero(dumbThreshold) < dumbThreshold.rows*dumbThreshold.cols/2) {
        preprocessed = 255 - preprocessed;
    }

    // another strategy from https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
    // to have uniform brightness, divide image by result of closure than subtract the result, divide it
    morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_BLACKHAT, structuringElement(100, cv::MORPH_ELLIPSE));
    cv::normalize(preprocessed, preprocessed, 0, 255, cv::NORM_MINMAX);

    Mat open = preprocessed;
    // shapes: MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    //cv::morphologyEx(open, open, cv::MorphTypes::MORPH_OPEN, structuringElement(4, cv::MORPH_ELLIPSE));
    // dilate to make text a bit more clear?

    Mat binarised;
    // create mask of what should be in the image
    //cv::normalize(open, open, 0, 255, cv::NORM_MINMAX);
    //int C = 0; // constant subtracted from calculated threshold value to obtain T(x, y)
    //cv::adaptiveThreshold(open, binarised, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 399, C);
    //cv::morphologyEx(binarised, binarised, cv::MorphTypes::MORPH_OPEN, structuringElement(2, cv::MORPH_ELLIPSE));
    cv::threshold(open, binarised, 0, 255, cv::THRESH_OTSU);

    //cv::normalize(morphInvOutput, invNormalised, 0, 255, cv::NORM_MINMAX);
    //cv::threshold(invNormalised, invOutput, 0, 255, cv::THRESH_OTSU);

    /*
    Mat threshold_mask;
    // keep the original intensity values by remultiplying this mask with the original image
    cv::multiply(preprocessed, threshold_mask, output);
    Mat ridges;
    cv::Ptr<cv::ximgproc::RidgeDetectionFilter> rdf = cv::ximgproc::RidgeDetectionFilter::create();
    rdf->getRidgeFilteredImage(output, ridges);
    */
    Mat labels;
    // stats is a 5 x nLabels Mat containing left, top, width, height, and area for each component (+ background)
    Mat stats;
    // centroids is a 2 x nLabels Mat containing the x, y coordinates of the centroid of each component
    Mat centroids;
    int nlabels = cv::connectedComponentsWithStats(binarised, labels, stats, centroids);

    vector<CComponent> allCCs;

    for (int label = 0; label < nlabels; ++label) {
        CComponent cc;
        cc.label = label;
        cc.left = stats.at<int>(label, cv::CC_STAT_LEFT);
        cc.top = stats.at<int>(label, cv::CC_STAT_TOP);
        cc.height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        cc.width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        cc.area = stats.at<int>(label, cv::CC_STAT_AREA);
        cc.centroidX = centroids.at<double>(label, 0);
        cc.centroidY = centroids.at<double>(label, 1);
        allCCs.push_back(cc);
    }

    showImage(image);
    showImage(open);
    showImage(binarised);

    vector<meanShift::Point<CComponent>> yCentroids;
    Mat allComponents = binarised.clone();
    int minArea = 200;
    for (CComponent& cc: allCCs) {
        if (cc.area >= minArea) {
            drawCC(allComponents, cc);
            //yCentroids.push_back(meanShift::Point {i, {centroidY}});
            // include height and width in clustering decision
            double ccHeight = static_cast<double>(cc.height);
            yCentroids.push_back(meanShift::Point<CComponent> {cc, {cc.centroidY, ccHeight}});
        }
    }
    showImage(allComponents);


    // TODO justify bandwidth parameter
    auto ccClusters = meanShift::cluster(yCentroids, 25);
    /*
    // show cluster modes
    for (Cluster& c : centroidClusters) {
        int mode = static_cast<int>(c.mode[0]);
        cv::line(allComponents, cv::Point(0, mode), cv::Point(allComponents.cols, mode), 255, thickness=3);
    }
    */

    // BELOW:
    // Essentially find outliers of row height for each row, and remove the corresponding bounding boxes/connected components

    /*
     * Within each cluster above, cluster again based on bounding box height, and keep only the largest cluster
     */
    vector<ccCluster> clustersByCentroid/*(centroidClusters.getSize())*/;
    for (ccCluster& ithCluster : ccClusters) {
        vector<meanShift::Point<CComponent>> ccLabelsInCluster/*(centroidClusters.getSize())*/;
        // use to find median of cluster height
        vector<int> heights;
        long sumHeight = 0;
        long sumSqHeight = 0;
        for (CComponent& cc : ithCluster.getData()) {
            sumHeight += cc.height;
            sumSqHeight += cc.height*cc.height;
            heights.push_back(cc.height);
            double heightD = static_cast<double>(cc.height);
            double widthD = static_cast<double>(cc.width);
            ccLabelsInCluster.push_back(meanShift::Point<CComponent> {cc, {heightD}});
        }
        // actually should use median height
        // TODO justify bandwidth parameter
        // should be scale invariant! -> mean, median, mode?

        size_t n = heights.size();
        double avgHeight = sumHeight/(double)n;
        double stddev = sqrt((1.0*sumSqHeight)/n - avgHeight*avgHeight);
        //double mean = (1.0*sumHeight)/heights.size();
        //double median = findMedian(heights);
        double bwMultiplier = 1.0;
        ccClusterList heightClusters = meanShift::cluster<CComponent>(ccLabelsInCluster, stddev*bwMultiplier);
        heightClusters.sortBySize();
        ccCluster biggest = heightClusters[0];
        clustersByCentroid.push_back(heightClusters[0]);
    }

    showCentroidClusters(binarised, clustersByCentroid);
    showRowBounds(binarised, clustersByCentroid);

    /*
     * For each row / centroid cluster:
     *      partition labels into sets where bounding boxes are 'close together':
     *      i.e. no two bounding boxes are farther apart than either of their widths
     *      // form a list of intervals using the x-coordinates of the bounding boxes
     *      // expand each interval by 50%?
     *      // return a list of each set of labels whose expanded bounding boxes/intervals overlap
     */

    vector<cv::Rect> overlappingCCRects;
    vector<size_t> rectsPerRow;
    for (ccCluster& c: clustersByCentroid) {
        vector<vector<Interval>> overlapping;
        vector<Interval> intervals;
        for (CComponent cc : c.getData()) {
            double left = static_cast<double>(cc.left);
            double width = static_cast<double>(cc.width);
            intervals.push_back(Interval(cc.label, left, left+width));
        }
        Interval::groupCloseIntervals(intervals, overlapping, 1.5);
        // make rects for each partition
        for (auto& partition : overlapping) {
            int minTop = image.rows; // >= anything inside image
            int minLeft = image.cols; // >= anything inside image
            int maxBottom = 0; // <= smaller than anything inside image
            int maxRight = 0; // <= smaller than anything inside image
            for (Interval& iv : partition) {
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
            overlappingCCRects.push_back(cv::Rect(minLeft, minTop, maxRight-minLeft, maxBottom-minTop));
        }
        rectsPerRow.push_back(overlapping.size());
    }
    showRects(binarised, overlappingCCRects);

    // estimate number of columns as median/mean of rects in each row?
    /*
     * find column separators as the vertical lines intersecting the least number of rectangles
     * there must be more rectangles lying between distinct column separators
     * than the number of rectangles that either one intersects
     */

    // TODO Contour / line detection
    return saveOrShowImage(binarised, argv[2]);
}
