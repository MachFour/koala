//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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

    Mat open = preprocessed;
    // shapes: MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    //cv::morphologyEx(open, open, cv::MorphTypes::MORPH_OPEN, structuringElement(10, cv::MORPH_CROSS));
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

    vector<cv::Rect> overlappingCCRects;

    tesseract::TessBaseAPI tesseractAPI;
    int tessStatus = tesseractAPI.Init("/usr/share/tessdata/", "eng", tesseract::OcrEngineMode::OEM_TESSERACT_ONLY);
    //int tessStatus = tesseractAPI.Init("/usr/share/tessdata/", "eng", tesseract::OcrEngineMode::OEM_LSTM_ONLY);
    if (tessStatus == -1) {
        fprintf(stderr, "Could not initialise tesseract API");
        return 1;
    }
    //tesseractAPI.ReadConfigFile();

    // tesseractAPI.SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK_VERT_TEXT);
    tesseractAPI.SetPageSegMode(tesseract::PageSegMode::PSM_RAW_LINE);
    tesseractAPI.SetSourceResolution(300);

    // NOTE this doesn't work when using the LSTM functionality of tesseract
    const char * whitelistChars = "1234567890%,-.<>abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const char * blacklistChars = "{}|";
    tesseractAPI.SetVariable("tessedit_char_whitelist", whitelistChars);
    tesseractAPI.SetVariable("tessedit_char_blacklist", blacklistChars);
    //cv::Ptr<cvTess> ocr = cvTess::create("/usr/share/", "eng", whitelistChars, cv::text::PSM_AUTO);

    for (ccCluster& c: clustersByCentroid) {
        vector<vector<Interval>> closeCCsInCluster;
        vector<Interval> intervals;
        for (CComponent cc : c.getData()) {
            double left = static_cast<double>(cc.left);
            double width = static_cast<double>(cc.width);
            intervals.push_back(Interval(cc.label, left, left+width));
        }
        Interval::groupCloseIntervals(intervals, closeCCsInCluster, 1.5);
        // TODO also make sure that they overlap vertically

        // make rects for each partition
        for (vector<Interval>& group : closeCCsInCluster) {
            cv::Rect expandedBB = findBoundingRect(group, allCCs, image.rows, image.cols);

            // simple filtering
            int area = expandedBB.height*expandedBB.width;
            if (area <= MIN_COMBINED_RECT_AREA || area > MAX_COMBINED_RECT_AREA) {
                continue;
            }
            overlappingCCRects.push_back(expandedBB);

            //tesseractAPI.SetRectangle(expandedBB.x, expandedBB.y, expandedBB.width, expandedBB.height);

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
                cv::bitwise_or(ccsInRect, 255, ccsInRect, /*mask=*/oneCC);
            }
            // clean it up a bit?
            cv::morphologyEx(ccsInRect, ccsInRect, cv::MorphTypes::MORPH_ERODE, structuringElement(3, cv::MORPH_ELLIPSE));
            cv::morphologyEx(ccsInRect, ccsInRect, cv::MorphTypes::MORPH_OPEN, structuringElement(2, 8, cv::MORPH_RECT));

            // invert for tesseract
            ccsInRect = 255-ccsInRect;
            // run Tesseract
            //vector<cv::Rect>   boxes;
            //vector<std::string> words;
            //vector<float>  confidences;
            int bytes_per_line = static_cast<int>(ccsInRect.step1()*ccsInRect.elemSize());
            tesseractAPI.SetImage(ccsInRect.data, ccsInRect.cols, ccsInRect.rows, /*bytes_per_pixel=*/1, bytes_per_line);
            //ocr->run(ccsInRect, outputText, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
            //tesseractAPI.Recognize(0);
            const char * out = tesseractAPI.GetUTF8Text();
            const char * tsvText = tesseractAPI.GetTSVText(0);
            const char * hocrText = tesseractAPI.GetHOCRText(0);
            const char * unlvText = tesseractAPI.GetUNLVText();

            /*
            Pix * img = tesseractAPI.GetInputImage();
            // create mat from img data. step parameter is number of bytes per row, wpl is (integer) words per line
            PIX * converted = pixConvert8To32(img);
            Mat tessPix(ccsInRect.rows, ccsInRect.cols, CV_8UC4, converted->data, sizeof(int)*converted->wpl);
            pixDestroy(&converted);
            showImage(tessPix);
            */

            /*
            std::string outputText(out);
            outputText.erase(remove(outputText.begin(), outputText.end(), '\n'), outputText.end());
            std::cout << "OCR output = \"" << outputText << "\" length = " << outputText.size() << std::endl;
            */
            printf("OCR text output: '%s'\n", out);
            printf("hOCR text: '%s'\n", hocrText);
            printf("tsv text: '%s'\n", tsvText);
            printf("UNLV text: '%s'\n", unlvText);
            delete[] out;
            delete[] hocrText;
            delete[] unlvText;
            delete[] tsvText;

            showImage(ccsInRect);
        }
    }
    showRects(binarised, overlappingCCRects);

    // estimate number of columns as median/mean of rects in each row?

    /*
     * TODO
     * find column separators as the vertical lines intersecting the least number of rectangles
     * there must be more rectangles lying between distinct column separators
     * than the number of rectangles that either one intersects
     */

    // TODO Contour / line detection
    //tesseractAPI.End();
    return saveOrShowImage(binarised, argv[2]);
}
