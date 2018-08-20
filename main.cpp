//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/ml.hpp>
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
            int area = w.getArea();
            if (area <= MIN_COMBINED_RECT_AREA || area > MAX_COMBINED_RECT_AREA) {
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
    }
    // this will count how many rects (from eligible rows) would be intersected by a cut at the given X coordinate
    //int * rectsCutByXCount = new int[image.cols];
    Mat rectsCutByXCount64F(image.cols, 1, CV_64FC1, cv::Scalar(0));
    Mat rectsCutByXCount(image.cols, 1, CV_8UC1, cv::Scalar(0));
    // this will count the total height of all rects (from eligible rows) that would be intersected by a cut at the given X coordinate
    Mat totalRectHeightByXCoord(image.cols, 1, CV_64FC1, cv::Scalar(0));
    for (auto& wordBB : wordBBsforColumnInference) {
        for (int j = wordBB.x; j < wordBB.x + wordBB.width; ++j) {
            rectsCutByXCount64F.at<double>(j)+= 1.0;
            totalRectHeightByXCoord.at<double>(j) += (double) wordBB.height;
        }
    }

    Mat smoothedRectCutDensity;
    {
        int blurSize = image.cols/32;
        if (blurSize % 2 == 0) {
            blurSize++;
        }
        cv::GaussianBlur(rectsCutByXCount64F, smoothedRectCutDensity, cv::Size(blurSize, blurSize), 0, 0);
        //cv::medianBlur(rectsCutByXCount64F, smoothedRectCutDensity, blurSize);
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
    /*
     * Fourier transform the smoothed Rect Cut density and find the strongest peak,
     * to estimate the number of columns
     */
    {
        // first remove the mean from the function to get rid of 'DC' energy;
        cv::Scalar avgVal = cv::mean(smoothedRectCutDensity);
        Mat m00 = smoothedRectCutDensity - avgVal;
        Mat m0, m1, m2, m3, m4, m5, m6;
        int N = cv::getOptimalDFTSize(smoothedRectCutDensity.rows);
        // pad with zeros
        // copyMakeBorder(src, dest, top, bottom, left, right, borderType, fillValue)
        cv::copyMakeBorder(m00, m0, 0, N - m00.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
        // complex-conjugate symmetry output
        cv::dft(m0, m1, cv::DFT_COMPLEX_OUTPUT);
        // get real and complex parts
        cv::extractChannel(m1, m2, 0);
        cv::extractChannel(m1, m3, 1);
        Mat phase(N, 1, CV_64FC1);
        for (int i = 0; i < N; ++i) {
            float im = static_cast<float>(m3.at<double>(i));
            float re = static_cast<float>(m2.at<double>(i));
            phase.at<double>(i, 0) = static_cast<double>(cv::fastAtan2(im, re));
        }
        // penalise phases far from odd multiples of 90
        Mat phasePenalty = 90 - cv::abs(90 - cv::abs(180-phase));
        // square each component
        cv::multiply(m2, m2, m4);
        cv::multiply(m3, m3, m5);
        cv::sqrt(m4 + m5, m6);

        //cv::mulSpectrums(m1, m1, m2, 0, false);
        //cv::abs()
        // square root and scale
        //cv::sqrt(m2, m3);
        cv::multiply(m6, 1.0/N, m6);
        // only take a portion of the spectrum (low frequencies, up to w = 2*pi/N*7
        // i.e. only look at periods down to 1/7 image width, as there are unlikely to be more than 7 columns
        int maxColumns = 6;
        Mat rectCutSpectrum = m6(cv::Rect(0, 0, 1, maxColumns+1));
        Mat rectCutPhase = phase(cv::Rect(0, 0, 1, maxColumns+1));
        Mat rectCutPhasePenalty = phasePenalty(cv::Rect(0, 0, 1, maxColumns+1));

        // find the max, ignoring the zero-frequency amplitude.
        //smoothedRectCutSpectrum.at<double>(0) = 0; // -> don't need this as we subtracted the mean above
        int maxLoc[2]; // x, y
        // minVal, maxVal, minLoc, maxLoc
        cv::minMaxIdx(rectCutSpectrum, NULL, NULL, NULL, maxLoc);
        cv::Ptr<Plot> plotSpectrum = makePlot(rectCutSpectrum, &image);
        cv::Ptr<Plot> plotPhase = makePlot(rectCutPhasePenalty, &image, cv::Scalar(255, 0, 255));
        Mat plotResultSpectrum, plotResultPhase; {
            plotSpectrum->render(plotResultSpectrum);
            plotPhase->render(plotResultPhase);
        }
        estimatedColumns = maxLoc[0];
        printf("Estimated columns: %d\n", estimatedColumns);
        showImage(0.3*plotResultSpectrum + 0.3*plotResultPhase + 0.4*rects);
    }

    // count number of peaks by sign counting, where the sign is taken relative to a threshold (3rd quartile?)
    {

    }

    {
        // now put horizontal centres of all wordBBs into a Mat and run kmeans
        Mat wordBBcentroidX((int)allWordBBs.size(), 1, CV_64FC1);
        printf("K means with centroid X values:\n");

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

        /*
        //Mat outputCentres;
        cv::kmeans(wordBBcentroidX, estimatedColumns, bestLabels, termCriteria, 3, cv::KMEANS_PP_CENTERS, outputCentres);
        */
        // apply column labels
        for (auto i = 0; i < (int) allWordBBs.size(); ++i) {
            int column = bestLabels.at<int>(i);
            printf("Label for wordBB %d: %d\n", i, bestLabels.at<int>(i));
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


    for (auto& row : rows) {
        for (wordBB& w : row) {
            const char * wordText = getText(tesseractAPI, w, /* printAndShow = */false);
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
