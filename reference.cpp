//
// Created by max on 9/11/18.
//

#include "reference.h"
#include "meanshift.h"
#include "Interval.h"
#include "ccomponent.h"
#include "plotutils.h"
#include "table.h"
#include "utils.h"
#include "ocrutils.h"
#include "wordBB.h"

//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

// android build uses different header files
#ifdef REFERENCE_ANDROID
#include <baseapi.h>
#include <allheaders.h>
#else
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#endif

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

/*
 * Helper functions
 */
int min(int a, int b) {
    return a <= b ? a : b;
}
int max(int a, int b) {
    return a >= b ? a : b;
}

// global parameters in terms of the size of the input image

// const int CENTROID_CLUSTER_BANDWIDTH = 20;
static int centroidClusterBandwidth(int w, int h) {
    return min(w, h)/75;
}


//const int MIN_CC_AREA = 80;
static int minCCArea(int w, int h) {
    return max(10, w*h/37500);
}

/*
const int MIN_COMBINED_RECT_AREA = 1000;
const int MAX_COMBINED_RECT_AREA = 2000*1500/4;
const int MIN_COMBINED_RECT_HEIGHT = 1500/150;
const int MAX_COMBINED_RECT_HEIGHT = 1500/4;
const int MIN_COMBINED_RECT_WIDTH = 2000/200;
const int MAX_COMBINED_RECT_WIDTH = 2000/4;
 */
static bool isPlausibleWordBBSize(const wordBB& w, int imgW, int imgH) {
    int imageArea = imgW*imgH;
    int wordArea = w.getArea();
    int wordW = w.width;
    int wordH = w.height;
    int minArea = imageArea/3000;
    int maxArea = imageArea/4;
    int minHeight = imgH/150;
    int maxHeight = imgH/4;
    int minWidth = imgW/200;
    //int maxWidth = imgW/4;

    return wordW >= minWidth && wordH >= minHeight && wordArea >= minArea
    /*&& wordW <= maxWidth */&& wordH <= maxHeight && wordArea <= maxArea;
}

/*
 * Stages of main processing
 */
static Mat preprocess(const Mat& image, bool batchMode);
static vector<vector<wordBB>> findWords(const Mat &binarised, bool batchMode);
static void classifyColumns(std::vector<wordBB>& words, int numColumns, int imageWidth, bool batchMode=true);



Table tableExtract(const Mat &image, tesseract::TessBaseAPI& tesseractAPI, cv::Mat * wordBBImg, bool batchMode) {
    Mat binarised = preprocess(image, batchMode);
    vector<vector<wordBB>> wordsByRow = findWords(binarised, batchMode);

    Mat rects = overlayWords(binarised, wordsByRow, false);
    if (!batchMode) {
        showImage(rects);
    }

    /*
     * find column separators as the vertical lines intersecting the least number of rectangles
     * there must be more rectangles lying between distinct column separators
     * than the number of rectangles that either one intersects
     */
    // estimate number of columns as median/mean of rects in each row?

    //unsigned char rectsPerRowIQR;
    unsigned char rectsPerRowQ1; // 1st quartile
    //unsigned char rectsPerRowQ2; // median
    //unsigned char rectsPerRowQ3; // 3rd quartile

    {
        Mat rectsPerRow;
        Mat sortedRectsPerRow;
        for (auto &row : wordsByRow) {
            auto s = static_cast<unsigned char>(row.size());
            if (s != 0) {
                rectsPerRow.push_back(s);
            }
        }
        cv::sort(rectsPerRow, sortedRectsPerRow, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
        //cv::integral(sortedRectsPerRow, sumRectsPerRow, sumSqRectsPerRow, CV_32S, CV_64F);
        auto n = rectsPerRow.size[0];
        rectsPerRowQ1 = sortedRectsPerRow.at<unsigned char>(n / 4);
        //rectsPerRowQ2 = sortedRectsPerRow.at<unsigned char>(n / 2);
        //rectsPerRowQ3 = sortedRectsPerRow.at<unsigned char>(n * 3 / 4);
        //rectsPerRowIQR = rectsPerRowQ3 - rectsPerRowQ1;
    }

    vector<wordBB> wordBBsforColumnInference;
    vector<wordBB> allWordBBs;
    {
        unsigned int rowNum = 0;
        for (auto &row : wordsByRow) {
            for (wordBB &w : row) {
                w.setRow(rowNum);
                if (row.size() >= rectsPerRowQ1) {
                    // otherwise too few rows for accurate column number estimation is likely to be faulty
                    wordBBsforColumnInference.push_back(w);
                }
                allWordBBs.push_back(w);
            }
            if (!row.empty()) {
                // don't increment rowNum if the row was empty
                rowNum++;
            }
        }
        if (!batchMode) {
            Mat rowRects = overlayWords(binarised, allWordBBs, false);
            showImage(rowRects);
        }
    }


    // this will count how many rects (from eligible rows) would be intersected by a cut at the given X coordinate
    //int * rectsCutByXCount = new int[image.cols];
    Mat rectsCutByXCount64F(image.cols, 1, CV_64FC1, cv::Scalar(0));
    // need 8UC1 for median blur
    Mat rectsCutByXCount(image.cols, 1, CV_8UC1, cv::Scalar(0));
    for (auto &wordBB : wordBBsforColumnInference) {
        for (int j = wordBB.x; j < wordBB.x + wordBB.width; ++j) {
            rectsCutByXCount64F.at<double>(j) += 1.0;
            rectsCutByXCount.at<unsigned char>(j) += 1;
        }
    }

    Mat smoothedRectCutDensity;
    Mat smoothedRectCutDensity32S;
    {
        int blurSize = image.cols / 8;
        if (blurSize % 2 == 0) {
            blurSize++;
        }
        {
            Mat tmp;
            cv::GaussianBlur(rectsCutByXCount64F, tmp, cv::Size(blurSize, blurSize), 0, 0);
            //cv::medianBlur(rectsCutByXCount, tmp, blurSize);
            tmp.convertTo(smoothedRectCutDensity32S, CV_32SC1);
            tmp.convertTo(smoothedRectCutDensity, CV_64FC1);
        }
        //cv::normalize(smoothedRectCutDensity)

#ifndef REFERENCE_ANDROID
        if (!batchMode) {
            cv::Ptr<Plot> plotCounts = makePlot(rectsCutByXCount64F, &image);
            cv::Ptr<Plot> plotSmoothedCounts = makePlot(smoothedRectCutDensity, &image);
            Mat plotResultCounts;
            {
                plotCounts->render(plotResultCounts);
            }
            Mat plotResultSmoothedCounts;
            {
                plotSmoothedCounts->render(plotResultSmoothedCounts);
            }

            showImage(0.5 * plotResultCounts + 0.5 * rects);
            showImage(0.5 * plotResultSmoothedCounts + 0.5 * rects);
        }
#endif // REFERENCE_ANDROID
    }

    unsigned int estimatedColumns;

    // count number of peaks by sign counting, where the sign is taken relative to a threshold (3rd quartile?)
    {
        Mat m0;
        cv::sort(smoothedRectCutDensity32S, m0, cv::SORT_ASCENDING | cv::SORT_EVERY_COLUMN);
        int n = smoothedRectCutDensity32S.rows;
        int q3 = m0.at<int>(n * 3 / 4);
        Mat m1 = smoothedRectCutDensity32S - q3;
        // count peaks by sign changes in m1
        unsigned int peaks = 0;
        bool inPeak = false;
        for (int i = 1; i < m1.rows; ++i) {
            bool atThreshold = (m1.at<int>(i) >= 0);
            if (atThreshold && !inPeak) {
                // just 'found' a new peak
                peaks++;
            }
            inPeak = atThreshold;
        }
        estimatedColumns = peaks;
        if (!batchMode) {
            printf("Estimated columns by sign counting: %u\n", estimatedColumns);
        }
#ifndef REFERENCE_ANDROID
        if (!batchMode) {
            cv::Ptr<Plot> plotThreshold = makePlot(smoothedRectCutDensity, &image);
            Mat plotResultThresh;
            {
                plotThreshold->render(plotResultThresh);
                // draw q3 as a line on the image
                // need to calculate its y coordinate, given that the plot's original height was equal to
                // max(smoothedCutRectDensity), but was rescaled to have height equal to the original image
                double maxVal;
                cv::minMaxIdx(smoothedRectCutDensity, nullptr, &maxVal);
                // subtract from 1.0 to get threshold referred to bottom of image, not top
                int thresholdYCoord = static_cast<int>((1.0 - q3 / maxVal) * plotResultThresh.rows);
                cv::rectangle(plotResultThresh, cv::Rect(0, thresholdYCoord, plotResultThresh.cols - 1, 1), 255, 5);
            }
            showImage(0.5 * plotResultThresh + 0.5 * rects);
        }
#endif // REFERENCE_ANDROID
    }

    classifyColumns(allWordBBs, estimatedColumns, image.cols, batchMode);

    // sort the wordBBs by row, then by column, then by x coordinate
    std::sort(allWordBBs.begin(), allWordBBs.end(), [](const wordBB &a, const wordBB &b) -> bool {
        // want to check whether a < b
        std::array<int, 3> aCoords {a.row(), a.col(), a.x};
        std::array<int, 3> bCoords {b.row(), b.col(), b.x};
        return std::lexicographical_compare(aCoords.begin(), aCoords.end(), bCoords.begin(), bCoords.end());
        /*if (a.row() != b.row()) {
            return a.row() < b.row();
        } else if (a.col() != b.col()) {
            return a.col() < b.col();
        } else {
            return a.x < b.x;
        }*/
    });

    // save intermediate processing result
    if (wordBBImg != nullptr) {
        *wordBBImg = overlayWords(binarised, allWordBBs, true);
    }

    Mat& imageForOcr = binarised; // use preprocessed to get Tesseract to do its own thresholding
    auto bytes_per_line = static_cast<int>(imageForOcr.step1() * imageForOcr.elemSize());
    tesseractAPI.SetImage(imageForOcr.data, imageForOcr.cols, imageForOcr.rows, /*bytes_per_pixel=*/1, bytes_per_line);
    tesseractAPI.SetSourceResolution(300);

    for (wordBB &w : allWordBBs) {
        w.text = getCleanedText(tesseractAPI, w);
    }

    // create table;
    Table t(estimatedColumns);
    {
        bool firstWord = true;
        int currentRow = 0;
        int currentColumn = 0;
        std::string cellText;
        for (const wordBB &w : allWordBBs) {
            if (w.row() != currentRow || w.col() != currentColumn || firstWord) {
                if (firstWord) {
                    firstWord = false;
                } else {
                    // set old cell
                    t.setColumnText(currentRow, currentColumn, cellText);
                }
                // start new cell;
                cellText = std::string(w.text);
                currentRow = w.row();
                currentColumn = w.col();
            } else {
                cellText.append(" ");
                cellText.append(w.text);
            }
        }
    }

    // TODO Contour / line detection
    //showImage(binarised);
    return t;
}

static void classifyColumns(vector<wordBB>& words, int estimatedColumns, int imageWidth, bool batchMode) {
    // now put horizontal centres of all wordBBs into a Mat and run kmeans
    Mat wordBBcentroidX((int) words.size(), 1, CV_64FC1);
    for (unsigned int i = 0; i < words.size(); ++i) {
        const wordBB& w = words[i];
        double centroidX = w.x + w.width / 2.0f;
        wordBBcentroidX.at<double>(i, 0) = centroidX;
    }

    if (!batchMode) {
        printf("Mixture model with %u components and centroid X values:\n", estimatedColumns);
        for (unsigned int i = 0; i < words.size(); ++i) {
            printf("%.1f, ", wordBBcentroidX.at<double>(i, 0));
        }
        printf("\n");
    }

    //cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, 0.001);

    /* Use EM Algorithm as slight generalisation of k-means, to allow different cluster sizes / column widths
     * initialise with means (column centres) spread uniformly across width of image.
     * (This wasn't easy to do with the k-means algorithm)
     */
    auto emAlg = cv::ml::EM::create();
    emAlg->setClustersNumber(estimatedColumns);
    //emAlg->setTermCriteria(termCriteria);
    // since we're in 1D, COV_MAT_DIAGONAL (separate sigma for each dim)
    // is equivalent to COV_MAT_SPHERICAL (all dims have same sigma)
    emAlg->setCovarianceMatrixType(cv::ml::EM::Types::COV_MAT_SPHERICAL);
    Mat initialMeans(estimatedColumns, 1, CV_64FC1);
    for (auto i = 0; i < estimatedColumns; ++i) {
        // want the centre (hence +0.5) of the ith column,
        // when image is divided into estimatedColumns parts of equal width
        initialMeans.at<double>(i, 0) = (i + 0.5) / estimatedColumns * imageWidth;
    }
    if (!batchMode) {
        for (auto i = 0; i < estimatedColumns; ++i) {
            printf("Mixture %d initial mean: %f\n", i, initialMeans.at<double>(i, 0));
        }
    }
    // cluster samples around initial means
    // don't provide initial covariance matrix estimates, or weights
    /*
     * signature:
     * trainE(InputArray samples, InputArray means0, InputArray covs0, InputArray weights0,
     *     OutputArray logLikelihoods, OutputArray labels, OutputArray probs)
     */
    Mat bestLabels;
    emAlg->trainE(wordBBcentroidX, initialMeans, cv::noArray(), cv::noArray(), cv::noArray(), bestLabels);

    Mat means = emAlg->getMeans();
    if (!batchMode) {
        for (auto i = 0; i < estimatedColumns; ++i) {
            printf("Mixture %d: mean=%f\n", i, means.at<double>(i));
        }
    }

    /*
    //Mat outputCentres;
    cv::kmeans(wordBBcentroidX, estimatedColumns, bestLabels, termCriteria, 3, cv::KMEANS_PP_CENTERS, outputCentres);
    */
    // apply column labels
    {
        /* Labels might not be in order of left-to-right columns, so we need to create this mapping */
        vector<int> labelOrder(estimatedColumns, 0);
        std::iota(labelOrder.begin(), labelOrder.end(), 0);
        std::sort(labelOrder.begin(), labelOrder.end(), [&means](int a, int b) -> bool {
            return means.at<double>(a) < means.at<double>(b);
        });
        const auto firstLabel = labelOrder.cbegin();
        const auto lastLabel = labelOrder.cend();
        for (auto i = 0; i < (int) words.size(); ++i) {
            auto currentLabel = bestLabels.at<int>(i);
            // TODO could be more efficient than std::find every time but it probably doesn't matter
            // find which column the label corresponds to
            // std::find returns an iterator, so the corresponding index is found by 'subtracting' the begin() iterator
            auto column = static_cast<unsigned int>(std::find(firstLabel, lastLabel, currentLabel) - firstLabel);
            words[i].setCol(column);
        }
    }
}


// do dumb threshold to figure out whether it's white on black or black on white text, and invert if necessary
// assumes that there is actually a clear majority of one over the other (i.e many more background pixels than foreground)
// ensure m has type CV_8UC1
static void makeWhiteTextOnBlack(Mat& m) {
    if (m.depth() != CV_8U) {
        throw std::invalid_argument("matrix must be CV_8U");
    }
    Mat dumbThreshold;
    cv::equalizeHist(m, dumbThreshold);
    //cv::threshold(dumbThreshold, dumbThreshold, 0, 255, cv::THRESH_OTSU);
    cv::adaptiveThreshold(dumbThreshold, dumbThreshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 399, 0);
    if (cv::countNonZero(dumbThreshold) >= dumbThreshold.rows * dumbThreshold.cols / 2) {
        // it seems like black text on white, invert it
        m = 255 - m;
    }
}

static Mat preprocess(const Mat& image, bool batchMode) {
    Mat grey;
    Mat preprocessed;
#ifdef REFERENCE_ANDROID
    // android images are RGBA after decoding
    cv::cvtColor(image, grey, CV_RGBA2GRAY);
#else
    grey = image;
#endif
    grey.convertTo(preprocessed, CV_8UC1);

    // do dumb threshold to figure out whether it's white on black or black on white text, and invert if necessary
    makeWhiteTextOnBlack(preprocessed);


    Mat linesRemoved;
    // remove large horizontal and vertical lines
    // use black hat instead of top hat since it's black on white
    cv::morphologyEx(preprocessed, linesRemoved, cv::MorphTypes::MORPH_TOPHAT, structuringElement(250, 5, cv::MORPH_RECT));
    cv::morphologyEx(linesRemoved, linesRemoved, cv::MorphTypes::MORPH_TOPHAT, structuringElement(5, 250, cv::MORPH_RECT));

    if (!batchMode) {
        showImage(image);
        showImage(preprocessed);
        showImage(linesRemoved);
    }

    Mat textEnhanced;
    // another strategy from https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
    // to have uniform brightness, divide image by result of closure than subtract the result, divide it
    int blackhatKSize = MIN(image.rows, image.cols)/15;
    cv::Mat sElement = structuringElement(blackhatKSize, cv::MORPH_ELLIPSE);
    morphologyEx(linesRemoved, textEnhanced, cv::MorphTypes::MORPH_TOPHAT, sElement);
    cv::normalize(textEnhanced, textEnhanced, 0, 255, cv::NORM_MINMAX);

    //clean it up a bit?
    // shapes: MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    //cv::morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_ERODE, structuringElement(2, cv::MORPH_ELLIPSE));
    //cv::morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_OPEN, structuringElement(1, 7, cv::MORPH_RECT));

    Mat binarised;
    //int C = 0; // constant subtracted from calculated threshold value to obtain T(x, y)
    //cv::adaptiveThreshold(open, binarised, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 399, C);
    //cv::morphologyEx(binarised, binarised, cv::MorphTypes::MORPH_OPEN, structuringElement(2, cv::MORPH_ELLIPSE));
    cv::threshold(textEnhanced, binarised, 0, 255, cv::THRESH_OTSU);

    if (!batchMode) {
        showImage(textEnhanced);
        showImage(binarised);
    }

    return binarised;

}

/*
 * Try to find 'words' in the image, just based on connected components:
 * Procedure:
 * 1. mean shift cluster the Y centroids and heights of the connected components,
 *      in order to group them by row (and to a lesser extent, height)
 * 2. Find the largst
 */
vector<vector<wordBB>> findWords(const Mat &binarised, bool batchMode) {
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

    vector<meanShift::Point<CComponent>> yCentroids;
    for (CComponent &cc: allCCs) {
        if (cc.area >= minCCArea(binarised.cols, binarised.rows)) {
            //yCentroids.push_back(meanShift::Point {i, {centroidY}});
            // include height and width in clustering decision
            auto ccHeight = static_cast<double>(cc.height);
            yCentroids.push_back(meanShift::Point<CComponent>{cc, {cc.centroidY, ccHeight}});
        }
    }
    if (!batchMode) {
        Mat allComponents = binarised.clone();
        for (CComponent &cc: allCCs) {
            if (cc.area >= minCCArea(binarised.cols, binarised.rows)) {
                drawCC(allComponents, cc);
            }
        }
        showImage(allComponents);
    }

    // TODO justify bandwidth parameter
    auto ccClusters = meanShift::cluster(yCentroids, centroidClusterBandwidth(binarised.rows, binarised.cols));
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
    for (ccCluster &ithCluster : ccClusters) {
        vector<meanShift::Point<CComponent>> ccLabelsInCluster/*(centroidClusters.getSize())*/;
        // use to find median of cluster height
        vector<int> heights;
        long sumHeight = 0;
        long sumSqHeight = 0;
        for (CComponent &cc : ithCluster.getData()) {
            sumHeight += cc.height;
            sumSqHeight += cc.height * cc.height;
            heights.push_back(cc.height);
            auto heightD = static_cast<double>(cc.height);
            //double widthD = static_cast<double>(cc.width);
            ccLabelsInCluster.emplace_back(meanShift::Point<CComponent>{cc, {heightD}});
        }
        // actually should use median height
        // TODO justify bandwidth parameter
        // should be scale invariant! -> mean, median, mode?

        auto n = heights.size();
        double avgHeight = sumHeight / (double) n;
        double stddev = sqrt((1.0 * sumSqHeight) / n - avgHeight * avgHeight);
        //double mean = (1.0*sumHeight)/heights.size();
        //double median = findMedian(heights);
        double bwMultiplier = 1.0;
        ccClusterList heightClusters = meanShift::cluster<CComponent>(ccLabelsInCluster, stddev * bwMultiplier);
        heightClusters.sortBySize();
        auto biggest = heightClusters[0];
        clustersByCentroid.push_back(biggest);
    }

    //showCentroidClusters(binarised, clustersByCentroid);
    //showRowBounds(binarised, clustersByCentroid);

    /*
     * For each row / centroid cluster:
     *      partition labels into sets where bounding boxes are 'close together':
     *      i.e. no two bounding boxes are farther apart than either of their widths
     *      // form a list of intervals using the x-coordinates of the bounding boxes
     *      // expand each interval by 50%?
     *      // return a list of each set of labels whose expanded bounding boxes/intervals overlap
     */



    // make 'rows' of 'words'
    vector<vector<wordBB>> rows;
    for (ccCluster &c: clustersByCentroid) {
        vector<vector<Interval>> closeCCsInCluster;
        vector<Interval> intervals;
        for (CComponent cc : c.getData()) {
            auto left = static_cast<double>(cc.left);
            auto width = static_cast<double>(cc.width);
            intervals.push_back(Interval(cc.label, left, left + width));
        }
        Interval::groupCloseIntervals(intervals, closeCCsInCluster, 1.5);
        // TODO prevent them from getting too big?
        // TODO also make sure that they overlap vertically

        // add new row
        vector<wordBB> row;
        rows.push_back(row);

        // make rects for each partition
        for (vector<Interval> &group : closeCCsInCluster) {
            wordBB w = wordBB(findBoundingRect(group, allCCs, binarised.rows, binarised.cols));
            // simple filtering
            // remove unlikely sized 'words'
            if (isPlausibleWordBBSize(w, binarised.cols, binarised.rows)) {
                rows.back().push_back(w);
            }
        }
    }
    return rows;
}
