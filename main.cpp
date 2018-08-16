//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
//#include <opencv2/ximgproc.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "reference.h"
#include "meanshift.h"
#include "Interval.h"
#include "ccomponent.h"

const int CENTROID_CLUSTER_BANDWIDTH = 20;
const int MIN_CC_AREA = 80;
const int MIN_COMBINED_RECT_AREA = 1000;
const int MAX_COMBINED_RECT_AREA = 2000*1500/4;

int tesseractInit(tesseract::TessBaseAPI&);
const char * getText(tesseract::TessBaseAPI&, cv::Rect, bool printAndShow=false);

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

    //clean it up a bit?
    // shapes: MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    //cv::morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_ERODE, structuringElement(2, cv::MORPH_ELLIPSE));
    //cv::morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_OPEN, structuringElement(1, 7, cv::MORPH_RECT));

    Mat open = preprocessed;

    Mat binarised;
    //int C = 0; // constant subtracted from calculated threshold value to obtain T(x, y)
    //cv::adaptiveThreshold(open, binarised, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 399, C);
    //cv::morphologyEx(binarised, binarised, cv::MorphTypes::MORPH_OPEN, structuringElement(2, cv::MORPH_ELLIPSE));
    cv::threshold(open, binarised, 0, 255, cv::THRESH_OTSU);

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
    for (CComponent& cc: allCCs) {
        if (cc.area >= MIN_CC_AREA) {
            drawCC(allComponents, cc);
            //yCentroids.push_back(meanShift::Point {i, {centroidY}});
            // include height and width in clustering decision
            double ccHeight = static_cast<double>(cc.height);
            yCentroids.push_back(meanShift::Point<CComponent> {cc, {cc.centroidY, ccHeight}});
        }
    }
    showImage(allComponents);


    // TODO justify bandwidth parameter
    auto ccClusters = meanShift::cluster(yCentroids, CENTROID_CLUSTER_BANDWIDTH);
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


    tesseract::TessBaseAPI tesseractAPI;
    if (tesseractInit(tesseractAPI) == -1) {
        fprintf(stderr, "Could not initialise tesseract API");
        return 1;
    }

    int bytes_per_line = static_cast<int>(preprocessed.step1()*preprocessed.elemSize());
    tesseractAPI.SetImage(preprocessed.data, preprocessed.cols, preprocessed.rows, /*bytes_per_pixel=*/1, bytes_per_line);
    tesseractAPI.SetSourceResolution(300);


    // make 'rows' of 'words'
    vector<vector<cv::Rect>> rows;
    for (ccCluster& c: clustersByCentroid) {
        vector<vector<Interval>> closeCCsInCluster;
        vector<Interval> intervals;
        for (CComponent cc : c.getData()) {
            double left = static_cast<double>(cc.left);
            double width = static_cast<double>(cc.width);
            intervals.push_back(Interval(cc.label, left, left + width));
        }
        Interval::groupCloseIntervals(intervals, closeCCsInCluster, 1.5);
        // TODO also make sure that they overlap vertically

        // add new row
        vector<cv::Rect> row;
        rows.push_back(row);

        // make rects for each partition
        for (vector<Interval> &group : closeCCsInCluster) {
            cv::Rect expandedBB = findBoundingRect(group, allCCs, image.rows, image.cols);
            // simple filtering
            int area = expandedBB.height * expandedBB.width;
            if (area <= MIN_COMBINED_RECT_AREA || area > MAX_COMBINED_RECT_AREA) {
                continue;
            } else {
                // add to rows
                rows.back().push_back(expandedBB);
            }
        }
    }
    Mat rects = overlayRects(binarised, rows);
    showImage(rects);

    /*
     * TODO
     * find column separators as the vertical lines intersecting the least number of rectangles
     * there must be more rectangles lying between distinct column separators
     * than the number of rectangles that either one intersects
     */
    // estimate number of columns as median/mean of rects in each row?


    unsigned char rectsPerRowIQR;
    unsigned char rectsPerRowQ1; // 1st quartile
    unsigned char rectsPerRowQ2; // median
    unsigned char rectsPerRowQ3; // 3rd quartile
    double avgRectsPerRow;
    double stddevRectsPerRow;

    {
        Mat rectsPerRow;
        Mat sortedRectsPerRow;
        for (auto &row : rows) {
            unsigned char s = static_cast<unsigned char>(row.size());
            if (s != 0) {
                rectsPerRow.push_back(s);
            }
        }
        cv::sort(rectsPerRow, sortedRectsPerRow, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);

        Mat sumRectsPerRow;
        Mat sumSqRectsPerRow;
        cv::integral(sortedRectsPerRow, sumRectsPerRow, sumSqRectsPerRow, CV_32S, CV_64F);
        int n = static_cast<int>(sumRectsPerRow.size[0]);
        rectsPerRowQ1 = sortedRectsPerRow.at<unsigned char>(n/4);
        rectsPerRowQ2 = sortedRectsPerRow.at<unsigned char>(n/2);
        rectsPerRowQ3 = sortedRectsPerRow.at<unsigned char>(n*3/4);
        rectsPerRowIQR = rectsPerRowQ3 - rectsPerRowQ1;
        avgRectsPerRow = sumRectsPerRow.at<int>(n-1)*1.0/n;
        double meanSqRectsPerRow = sumSqRectsPerRow.at<double>(n-1)/n;
        stddevRectsPerRow = sqrt(meanSqRectsPerRow - avgRectsPerRow*avgRectsPerRow);
    }
    // alt: find median

    // this will count how many rects (from eligible rows) would be intersected by a cut at the given X coordinate
    //int * rectsCutByXCount = new int[image.cols];
    Mat rectsCutByXCount(image.cols, 1, CV_64FC1, cv::Scalar(0));
    // this will count the total height of all rects (from eligible rows) that would be intersected by a cut at the given X coordinate
    Mat totalRectHeightByXCoord(image.cols, 1, CV_64FC1, cv::Scalar(0));
    for (auto& row : rows) {
        if (row.size() < rectsPerRowQ1) {
            continue;
        }
        for (cv::Rect rect : row) {
            for (int j = rect.x; j < rect.x + rect.width; ++j) {
                rectsCutByXCount.at<double>(j)+= 1.0;
                totalRectHeightByXCoord.at<double>(j) += (double) rect.height;
            }
        }
    }

    Mat smoothedRectsCutByXCount;
    {
        int blurSize = image.cols/8;
        if (blurSize % 2 == 0) {
            blurSize++;
        }
        cv::GaussianBlur(rectsCutByXCount, smoothedRectsCutByXCount, cv::Size(blurSize, blurSize), 0, 0);
    }

    // now find minima
    int howManyMinima = 5;
    // findpeaks??
    // todo plotCounts and find a useful benchmark?

    {
        cv::Ptr<cv::plot::Plot2d> plotCounts = cv::plot::Plot2d::create(rectsCutByXCount);
        plotCounts->setNeedPlotLine(true);
        plotCounts->setShowGrid(false);
        plotCounts->setPlotLineWidth(7);
        plotCounts->setPlotSize(image.cols, image.rows);
        plotCounts->setInvertOrientation(true);

        cv::Ptr<cv::plot::Plot2d> smoothedPlotCounts = cv::plot::Plot2d::create(smoothedRectsCutByXCount);
        smoothedPlotCounts->setNeedPlotLine(true);
        smoothedPlotCounts->setShowGrid(false);
        smoothedPlotCounts->setPlotLineWidth(7);
        smoothedPlotCounts->setPlotLineColor(cv::Scalar(0, 255, 0));
        smoothedPlotCounts->setPlotSize(image.cols, image.rows);
        smoothedPlotCounts->setInvertOrientation(true);

        Mat plotResultCounts;
        Mat plotResultSmoothedCounts;

        plotCounts->render(plotResultCounts);
        smoothedPlotCounts->render(plotResultSmoothedCounts);
        showImage(0.5*plotResultCounts + 0.5*rects);
        showImage(0.5*plotResultSmoothedCounts + 0.5*rects);
    }

    // TODO find peaks/troughs (minimum values). Make sure that two peaks are 'distinct' enough in which boxes they separate
    // TODO once columns are decided, then classify words on either side by centroid location

    for (auto& row : rows) {
        for (cv::Rect wordBB : row) {
            const char * wordText = getText(tesseractAPI, wordBB, /* printAndShow = */false);
            delete[] wordText;
        }
    }

    // TODO Contour / line detection
    tesseractAPI.End();
    return saveOrShowImage(binarised, argv[2]);
}

int tesseractInit(tesseract::TessBaseAPI& baseAPI) {
    int tessStatus = baseAPI.Init("/usr/share/tessdata/", "eng", tesseract::OcrEngineMode::OEM_TESSERACT_ONLY);
    //int tessStatus = tesseractAPI.Init("/usr/share/tessdata/", "eng", tesseract::OcrEngineMode::OEM_LSTM_ONLY);
    if (tessStatus == -1) {
        return tessStatus;
    }
    //tesseractAPI.ReadConfigFile();

    // tesseractAPI.SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK_VERT_TEXT);
    baseAPI.SetPageSegMode(tesseract::PageSegMode::PSM_RAW_LINE);

    // NOTE this doesn't work when using the LSTM functionality of tesseract
    const char * whitelistChars = "1234567890%,-.<>abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const char * blacklistChars = "{}|";
    baseAPI.SetVariable("tessedit_char_whitelist", whitelistChars);
    baseAPI.SetVariable("tessedit_char_blacklist", blacklistChars);
    //cv::Ptr<cvTess> ocr = cvTess::create("/usr/share/", "eng", whitelistChars, cv::text::PSM_AUTO);
    return 0;

}

// text must be delete[]d after use.
// API must be initialised with image
const char * getText(tesseract::TessBaseAPI& tesseractAPI, cv::Rect roi, bool printAndShow) {
    tesseractAPI.SetRectangle(roi.x, roi.y, roi.width, roi.height);

    /*
    // now combine each set of close CCs into an image
    Mat binarisedCCroi(labels, expandedBB); // view of label matrix corresponding to current region of interest
    Mat ccsInRect(expandedBB.height, expandedBB.width, CV_8UC1); // this will hold our connected component image
    ccsInRect = 0; // without this, the image gets corrupted with crap
    //printf("grouped CCs inside rect: x=%d, y=%d, w=%d, ht=%d\n", expandedBB.x, expandedBB.y, expandedBB.width, expandedBB.height);
    for (Interval& iv : group) {
        // TODO avoid indexing back into global list
        Mat oneCC(binarisedCCroi.rows, binarisedCCroi.cols, CV_8UC1);
        cv::compare(binarisedCCroi, iv.getLabel(), oneCC, cv::CMP_EQ);
        //cv::bitwise_or(ccsInRect, oneCC, ccsInRect);
        // alternatively, use oneCC as mask
        cv::bitwise_or(ccsInRect, 255, ccsInRect, oneCC);
    }
    // clean it up a bit?
    cv::morphologyEx(ccsInRect, ccsInRect, cv::MorphTypes::MORPH_ERODE, structuringElement(3, cv::MORPH_ELLIPSE));
    cv::morphologyEx(ccsInRect, ccsInRect, cv::MorphTypes::MORPH_OPEN, structuringElement(2, 8, cv::MORPH_RECT));
    */

    // invert for tesseract
    //ccsInRect = 255-ccsInRect;
    // run Tesseract
    //vector<cv::Rect>   boxes;
    //vector<std::string> words;
    //vector<float>  confidences;
    //ocr->run(ccsInRect, outputText, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
    //tesseractAPI.Recognize(0);
    const char * out = tesseractAPI.GetUTF8Text();
    //const char * tsvText = tesseractAPI.GetTSVText(0);
    //const char * hocrText = tesseractAPI.GetHOCRText(0);
    //const char * unlvText = tesseractAPI.GetUNLVText();

    if (!printAndShow) {
        return out;
    }

    Pix * img = tesseractAPI.GetThresholdedImage();
    // create mat from img data. step parameter is number of bytes per row, wpl is (integer) words per line

    //PIX * converted = pixConvert1To32(nullptr, img, 0, 255);
    //Mat tessPix(converted->h, converted->w, CV_8UC4, converted->data, sizeof(int)*converted->wpl);
    Mat tessPix = matFromPix1(img);
    pixDestroy(&img);
    //cv::cvtColor(tessPix, tessPix, CV_BGRA2GRAY);
    //cv::rectangle(tessPix, expandedBB, cv::Scalar(255, 255, 255), 4);

    /*
    std::string outputText(out);
    outputText.erase(remove(outputText.begin(), outputText.end(), '\n'), outputText.end());
    std::cout << "OCR output = \"" << outputText << "\" length = " << outputText.size() << std::endl;
    */
    printf("OCR text output: '%s'\n", out);
    // output hocr
    //char hocrFileName[30] {'\0'};
    //snprintf(hocrFileName, 30, "hocr-rect-%d-%d-%d-%d.txt", expandedBB.x, expandedBB.y, expandedBB.width, expandedBB.height);
    //std::ofstream hocrFile(hocrFileName);
    //hocrFile << hocrText;
    //hocrFile.close();
    //printf("hOCR text written to %s\n", hocrFileName);
    //printf("tsv text: '%s'\n", tsvText);
    //printf("UNLV text: '%s'\n", unlvText);
    //delete[] out;
    //delete[] hocrText;
    //delete[] unlvText;
    //delete[] tsvText;

    showImage(tessPix);
    //showImage(ccsInRect);

    return out;
}
