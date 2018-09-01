//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/ml.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "reference.h"
#include "meanshift.h"
#include "Interval.h"
#include "ccomponent.h"

const int CENTROID_CLUSTER_BANDWIDTH = 20;
const int MIN_CC_AREA = 80;
const int MIN_COMBINED_RECT_AREA = 1000;
const int MAX_COMBINED_RECT_AREA = 2000*1500/4;
const int MIN_COMBINED_RECT_HEIGHT = 1500/150;
const int MAX_COMBINED_RECT_HEIGHT = 1500/4;
const int MIN_COMBINED_RECT_WIDTH = 2000/200;
const int MAX_COMBINED_RECT_WIDTH = 2000/4;

int tesseractInit(tesseract::TessBaseAPI&);
const char * getText(tesseract::TessBaseAPI&, wordBB, bool printAndShow=false);

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

    //showImage(image);
    //showImage(open);
    //showImage(binarised);

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
    //showImage(allComponents);


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


    tesseract::TessBaseAPI tesseractAPI;
    if (tesseractInit(tesseractAPI) == -1) {
        fprintf(stderr, "Could not initialise tesseract API");
        return 1;
    }

    int bytes_per_line = static_cast<int>(preprocessed.step1()*preprocessed.elemSize());
    tesseractAPI.SetImage(preprocessed.data, preprocessed.cols, preprocessed.rows, /*bytes_per_pixel=*/1, bytes_per_line);
    tesseractAPI.SetSourceResolution(300);


    // make 'rows' of 'words'
    vector<vector<wordBB>> rows;
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
        vector<wordBB> row;
        rows.push_back(row);

        // make rects for each partition
        for (vector<Interval> &group : closeCCsInCluster) {
            wordBB w = wordBB(findBoundingRect(group, allCCs, image.rows, image.cols));
            // simple filtering
            // remove unlikely sized 'words'
            int area = w.getArea();
            int height = w.height;
            int width = w.width;
            if (height <= MIN_COMBINED_RECT_HEIGHT || height > MAX_COMBINED_RECT_HEIGHT
                    || width <= MIN_COMBINED_RECT_WIDTH //|| width > MAX_COMBINED_RECT_WIDTH
                    || area <= MIN_COMBINED_RECT_AREA || area > MAX_COMBINED_RECT_AREA) {
                continue;
            } else {
                // add to rows
                rows.back().push_back(w);
            }
        }
    }

    Mat rects = overlayWords(binarised, rows, false);
    showImage(rects);

    /*
     * find column separators as the vertical lines intersecting the least number of rectangles
     * there must be more rectangles lying between distinct column separators
     * than the number of rectangles that either one intersects
     */
    // estimate number of columns as median/mean of rects in each row?


    unsigned char rectsPerRowIQR;
    unsigned char rectsPerRowQ1; // 1st quartile
    unsigned char rectsPerRowQ2; // median
    unsigned char rectsPerRowQ3; // 3rd quartile

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
        //cv::integral(sortedRectsPerRow, sumRectsPerRow, sumSqRectsPerRow, CV_32S, CV_64F);
        int n = static_cast<int>(rectsPerRow.size[0]);
        rectsPerRowQ1 = sortedRectsPerRow.at<unsigned char>(n/4);
        rectsPerRowQ2 = sortedRectsPerRow.at<unsigned char>(n/2);
        rectsPerRowQ3 = sortedRectsPerRow.at<unsigned char>(n*3/4);
        rectsPerRowIQR = rectsPerRowQ3 - rectsPerRowQ1;
    }

    vector<wordBB> wordBBsforColumnInference;
    vector<wordBB> allWordBBs;
    {
        int rowNum = 0;
        for (auto &row : rows) {
            if (row.size() == 0) {
                continue;
            }
            for (wordBB& w : row) {
                w.row = rowNum;

                allWordBBs.push_back(w);
                if (row.size() >= rectsPerRowQ1) {
                    // otherwise too few rows for accurate column number estimation is likely to be faulty
                    wordBBsforColumnInference.push_back(w);
                }
            }
            rowNum++;
        }
        // TODO colour by row
        Mat rowRects = overlayWords(binarised, allWordBBs, false);
        showImage(rowRects);
    }


    // this will count how many rects (from eligible rows) would be intersected by a cut at the given X coordinate
    //int * rectsCutByXCount = new int[image.cols];
    Mat rectsCutByXCount64F(image.cols, 1, CV_64FC1, cv::Scalar(0));
    // need 8UC1 for median blur
    Mat rectsCutByXCount(image.cols, 1, CV_8UC1, cv::Scalar(0));
    for (auto& wordBB : wordBBsforColumnInference) {
        for (int j = wordBB.x; j < wordBB.x + wordBB.width; ++j) {
            rectsCutByXCount64F.at<double>(j) += 1.0;
            rectsCutByXCount.at<unsigned char>(j) += 1;
        }
    }

    Mat smoothedRectCutDensity;
    Mat smoothedRectCutDensity32S;
    {
        int blurSize = image.cols/8;
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

        cv::Ptr<Plot> plotCounts = makePlot(rectsCutByXCount64F, &image);
        cv::Ptr<Plot> plotSmoothedCounts = makePlot(smoothedRectCutDensity, &image);
        Mat plotResultCounts; {
            plotCounts->render(plotResultCounts);
        }
        Mat plotResultSmoothedCounts; {
            plotSmoothedCounts->render(plotResultSmoothedCounts);
        }

        showImage(0.5*plotResultCounts + 0.5*rects);
        showImage(0.5*plotResultSmoothedCounts + 0.5*rects);
    }

    int estimatedColumns;

    // count number of peaks by sign counting, where the sign is taken relative to a threshold (3rd quartile?)
    {
        Mat m0;
        cv::sort(smoothedRectCutDensity32S, m0, cv::SORT_ASCENDING | cv::SORT_EVERY_COLUMN);
        int n = smoothedRectCutDensity32S.rows;
        int q3 = m0.at<int>(n*3/4);
        Mat m1 = smoothedRectCutDensity32S - q3;
        // count peaks by sign changes in m1
        int peaks = 0;
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
        printf("Estimated columns by sign counting: %d\n", estimatedColumns);
        cv::Ptr<Plot> plotThreshold = makePlot(smoothedRectCutDensity, &image);
        Mat plotResultThresh; {
            plotThreshold->render(plotResultThresh);
            // draw q3 as a line on the image
            // need to calculate its y coordinate, given that the plot's original height was equal to
            // max(smoothedCutRectDensity), but was rescaled to have height equal to the original image
            double maxVal;
            cv::minMaxIdx(smoothedRectCutDensity, NULL, &maxVal);
            // subtract from 1.0 to get threshold referred to bottom of image, not top
            int thresholdYCoord = static_cast<int>((1.0 - q3/maxVal)*plotResultThresh.rows);
            cv::rectangle(plotResultThresh, cv::Rect(0, thresholdYCoord, plotResultThresh.cols-1, 1), 255, 5);
        }
        showImage(0.5*plotResultThresh + 0.5*rects);
    }

    {
        // now put horizontal centres of all wordBBs into a Mat and run kmeans
        Mat wordBBcentroidX((int)allWordBBs.size(), 1, CV_64FC1);
        printf("Mixture model with %d components and centroid X values:\n", estimatedColumns);

        for (auto i = 0; i < allWordBBs.size(); ++i) {
            wordBB& w = allWordBBs[i];
            double centroidX = w.x+w.width/2.0f;
            wordBBcentroidX.at<double>(i, 0) = centroidX;
            printf("%f, ", centroidX);
        }
        printf("\n");
        cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, 0.001);

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
            double mixtureMean = (i+0.5)/estimatedColumns*image.cols;
            printf("Mixture %d initial mean: %f\n", i, mixtureMean);
            initialMeans.at<double>(i, 0) = mixtureMean;
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
        for (int i = 0; i < estimatedColumns; ++i) {
            printf("Mixture %d: mean=%f\n", i, means.at<double>(i));
        }
        /* Labels might not be in order of left-to-right columns, so we need to create this mapping */
        vector<int> mixtureLabelToColumn;
        for (int i = 0; i < estimatedColumns; ++i) {
            mixtureLabelToColumn.push_back(i);
        }
        std::sort(mixtureLabelToColumn.begin(), mixtureLabelToColumn.end(), [&means](int a, int b) -> bool {
            return means.at<double>(a) < means.at<double>(b);
        });

        /*
        //Mat outputCentres;
        cv::kmeans(wordBBcentroidX, estimatedColumns, bestLabels, termCriteria, 3, cv::KMEANS_PP_CENTERS, outputCentres);
        */
        // apply column labels
        for (auto i = 0; i < (int) allWordBBs.size(); ++i) {
            int column = mixtureLabelToColumn[bestLabels.at<int>(i)];
            //printf("Label for wordBB %d: %d\n", i, bestLabels.at<int>(i));
            allWordBBs[i].column = column;
        }
    }

    // TODO find peaks/troughs (minimum values). Make sure that two peaks are 'distinct' enough in which boxes they separate
    // TODO once columns are decided, then classify words on either side by centroid location

    {
        Mat rects2 = overlayWords(binarised, allWordBBs, true);
        showImage(rects2);
    }

    //  now find peak of spectrum and use that for kmeans

    // sort by row, then by column, then by x coordinate
    std::sort(allWordBBs.begin(), allWordBBs.end(), [](const wordBB& a, const wordBB& b) -> bool {
        // want to check whether a < b
        if (a.row != b.row) {
            return a.row < b.row;
        } else if (a.column != b.column) {
            return a.column < b.column;
        } else {
            return a.x < b.x;
        }
    });

    for (wordBB& w : allWordBBs) {
        const char * wordText = getText(tesseractAPI, w, /* printAndShow = */false);
        w.text = std::string(wordText);
        // remove newlines
        w.text.erase(remove(w.text.begin(), w.text.end(), '\n'), w.text.end());
        w.text.shrink_to_fit();
        printf("Text found: '%s'\n", w.text.data());
        delete[] wordText;
    }

    // print to console
    {
        printf("\n\n");
        int currentRow = 0;
        int currentColumn = 0;
        int currentColumnChars = 0;
        for (const wordBB& w : allWordBBs) {
            if (currentRow != w.row) {
                printf("\n");
                currentRow = w.row;
                // new column might not be zero
                currentColumnChars = 0;
                currentColumn = 0;
            }
            if (currentColumn != w.column) {
                assert (currentColumn < w.column); // should always be true after sorting
                for (int col = currentColumn; col < w.column; ++col) {
                    // end column, pad out to 40 chars
                    for (int i = 0; i + currentColumnChars < 30; ++i) {
                        putchar(' ');
                    }
                    putchar('|');
                    putchar(' ');
                    currentColumnChars = 0;
                }
                currentColumn = w.column;
            }
            currentColumnChars += w.text.length() + 1;
            printf("%s ", w.text.data());
        }
        printf("\n");
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
const char * getText(tesseract::TessBaseAPI& tesseractAPI, wordBB roi, bool printAndShow) {
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
