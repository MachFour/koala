//
// Created by max on 9/11/18.
//

#include "reference.h"
#include "meanshift.h"
#include "Interval.h"
#include "ccomponent.h"
#include "plotutils.h"
#include "table.h"
#include "helpers.h"
#include "matutils.h"
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
#include <vector>


using cv::Mat;
using std::vector;

// global parameters in terms of the size of the input image

// const int CENTROID_CLUSTER_BANDWIDTH = 20;
static double rowClusterBandwidth(int h) {
    return h/80.0;
}
static double heightClusterBandwidth(int h) {
    return 1.5*rowClusterBandwidth(h);
}


//const int MIN_CC_AREA = 80;
// note that the area of a CC is given by its white pixels, not the area of its bounding box
static int minCCArea(int w, int h) {
    return std::max(10, w*h/37500);
}
static int minCCBoxArea(int w, int h) {
    return 2*minCCArea(w, h);
}

static int maxCCBoxArea(int w, int h) {
    return w*h/20;
}

// Returns whether x is in the closed interval [a, b]
// make sure that a <= b!
template<typename T>
static bool inInterval(T x, T a, T b) {
    return x >= a && x <= b;
}

/*
const int MIN_COMBINED_RECT_AREA = 1000;
const int MAX_COMBINED_RECT_AREA = 2000*1500/4;
const int MIN_COMBINED_RECT_HEIGHT = 1500/150;
const int MAX_COMBINED_RECT_HEIGHT = 1500/4;
const int MIN_COMBINED_RECT_WIDTH = 2000/200;
const int MAX_COMBINED_RECT_WIDTH = 2000/4;
 */
const double MAX_CC_ASPECT_RATIO = 9;
const double MIN_CC_ASPECT_RATIO = 0.05;
const double MAX_RECT_ASPECT_RATIO = 20;
const double MIN_RECT_ASPECT_RATIO = 0.05;

/*
 * Connected component and combined CC / wordBB size
 */
static bool isPlausibleCCSize(const CC& c, int imgW, int imgH) {
    return c.area >= minCCArea(imgW, imgH) &&
           c.boxArea() >= minCCBoxArea(imgW, imgH) &&
           c.boxArea() <= maxCCBoxArea(imgW, imgH) &&
           inInterval(c.boxAspectRatio(), MIN_CC_ASPECT_RATIO, MAX_CC_ASPECT_RATIO);
}

static bool isPlausibleWordBBSize(const wordBB& w, int imgW, int imgH) {
    int minArea = 0; //imageArea/3000;
    int maxArea = imgW*imgH/4;
    int minHeight = imgH/150;
    int maxHeight = imgH/4;
    int minWidth = imgW/200;
    int maxWidth = imgW*3/4;

    return inInterval(w.width(), minWidth, maxWidth)
           && inInterval(w.height(), minHeight, maxHeight)
           && inInterval(w.boxArea(), minArea, maxArea)
           && inInterval(w.boxAspectRatio(), MIN_RECT_ASPECT_RATIO, MAX_RECT_ASPECT_RATIO);
}

/*
 * Stages of main processing
 */
static Mat preprocess(const Mat&, bool);
static vector<vector<wordBB>> findWords(const Mat&, bool);
static int estimateNumberOfColumns(const vector<wordBB>& wordBBs, int width, const Mat& rects, bool batchMode);
static void classifyColumns(std::vector<wordBB>&, int, const Mat& rects, bool batchMode=true);
static void doOcr(const Mat&, tesseract::TessBaseAPI&, vector<wordBB>&);

Table tableExtract(const Mat &image, tesseract::TessBaseAPI& tesseractAPI, cv::Mat * wordBBImg, bool batchMode) {

    Mat grey;
#ifdef REFERENCE_ANDROID
    // android images are RGBA after decoding
    cv::cvtColor(image, grey, CV_RGBA2GRAY);
#else
    grey = image;
#endif
    Mat grey8;
    grey.convertTo(grey8, CV_8UC1);

    Mat preprocessed = preprocess(grey8, batchMode);
    Mat binarised;
    {
        // gaussian blur
        Mat preprocessedF = eightBitToFloat(preprocessed);
        Mat blurredF;
        cv::GaussianBlur(preprocessedF, blurredF, cv::Size(5, 5), 0);
        Mat blurred = floatToEightBit(blurredF);
        //cv::threshold(blurred, binarised, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::threshold(preprocessed, binarised, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (!batchMode) {
            showImage(binarised, "binarised");
        }
    }

    vector<vector<wordBB>> wordsByRow = findWords(binarised, batchMode);

    Mat rects = overlayWords(binarised, wordsByRow, false);
    if (!batchMode) {
        showImage(rects, "combined rects");
    }

    /*
     * find column separators as the vertical lines intersecting the least number of rectangles
     * there must be more rectangles lying between distinct column separators
     * than the number of rectangles that either one intersects
     */

    // only use rows with more word boxes than this to estimate column count
    // set to some quantiles

    size_t minRectsPerRow;
    size_t maxRectsPerRow;
    {
        std::vector<size_t> rectsPerRow;
        for (auto &row : wordsByRow) {
            if (!row.empty()) {
                rectsPerRow.push_back(row.size());
            }
        }
        if (!rectsPerRow.empty()) {
            std::sort(rectsPerRow.begin(), rectsPerRow.end());
            auto n = rectsPerRow.size(); // guaranteed n > 0
            minRectsPerRow = rectsPerRow[n/4];
            maxRectsPerRow = rectsPerRow[n*4/5];
        } else {
            minRectsPerRow = 0;
            maxRectsPerRow = (size_t) -1; // largest size
            fprintf(stderr, "tableExtract(): no nonempty rows!!\n");
        }
    }

    vector<wordBB> wordBBsforColumnInference;
    vector<wordBB> allWordBBs;
    {
        unsigned int rowNum = 0;
        for (auto &row : wordsByRow) {
            for (wordBB &w : row) {
                w.setRow(rowNum);
                if (row.size() >= minRectsPerRow && row.size() <= maxRectsPerRow) {
                    // otherwise it will probably screw up column number estimation

                    // add a little but of overlap when we do the row counts
                    wordBB toExpand(w);
                    toExpand.expandWidthPx(15);
                    toExpand.constrain(0, 0, binarised.cols, binarised.rows);
                    wordBBsforColumnInference.push_back(toExpand);
                }
                allWordBBs.push_back(w);
            }
            if (!row.empty()) {
                // don't increment rowNum if the row was empty
                rowNum++;
            }
        }
    }

    Mat allRowRects = overlayWords(binarised, allWordBBs, true);
    Mat columnRects = overlayWords(binarised, wordBBsforColumnInference, true);
    if (!batchMode) {
        showImage(allRowRects, "all combined WordBBs");
        showImage(columnRects, "wordBBs for column Inference");
    }

    int estimatedColumns = estimateNumberOfColumns(wordBBsforColumnInference, image.cols, columnRects, batchMode);
    classifyColumns(allWordBBs, estimatedColumns, columnRects, batchMode);

    // sort the wordBBs by row, then by column, then by x coordinate
    std::sort(allWordBBs.begin(), allWordBBs.end(), [](const wordBB &a, const wordBB &b) -> bool {
        // sort predicate returns true if a < b
        std::array<int, 3> aCoords {a.row(), a.col(), a.left()};
        std::array<int, 3> bCoords {b.row(), b.col(), b.left()};
        return std::lexicographical_compare(aCoords.begin(), aCoords.end(), bCoords.begin(), bCoords.end());
    });

    Mat classifiedWordBBs = overlayWords(binarised, allWordBBs, true);
    if (!batchMode) {
        showImage(classifiedWordBBs, "classified wordBBs");
    }

    vector<wordBB> combinedWordBBs;
    {
        vector<wordBB> wordsInCurrentCell;
        bool firstWord = true;
        int currentRow = 0;
        int currentColumn = 0;
        for (const wordBB &w : allWordBBs) {
            if (w.row() != currentRow || w.col() != currentColumn || firstWord) {
                if (firstWord) {
                    firstWord = false;
                } else {
                    auto combined = wordBB::combineAll(wordsInCurrentCell);
                    combined.expandMinOf(10, 0.2);
                    combined.constrain(0, 0, image.cols, image.rows);
                    combined.setCol(currentColumn);
                    combined.setRow(currentRow);
                    combinedWordBBs.push_back(combined);
                }
                // start new cell;
                wordsInCurrentCell.clear();
                wordsInCurrentCell.push_back(w);
                currentRow = w.row();
                currentColumn = w.col();
            } else {
                wordsInCurrentCell.push_back(w);
            }
        }
    }

    Mat finalWordBBs = overlayWords(binarised, combinedWordBBs, true);
    if (!batchMode) {
        showImage(finalWordBBs, "combinedWordBBs");
    }
    if (wordBBImg != nullptr) {
        // save intermediate processing result
        *wordBBImg = finalWordBBs;
    }

    // run OCR on the preprocessed image
    doOcr(invert(preprocessed), tesseractAPI, combinedWordBBs);

    Table t(estimatedColumns);
    // now we can assume there is only one wordBB per row and column
    for (const wordBB &w : combinedWordBBs) {
        t.setColumnText(w.row(), w.col(), w.text());
    }

    // TODO Contour / line detection
    //showImage(binarised);
    return t;
}
static cv::Mat preprocess(const cv::Mat& grey8, bool batchMode) {
    using cv::Mat;

    // do dumb threshold to figure out whether it's white on black or black on white text, and invert if necessary
    Mat whiteOnBlack = isWhiteTextOnBlack(grey8) ? grey8 : invert(grey8);

    const auto openingKsize = std::max(whiteOnBlack.rows, whiteOnBlack.cols)/30;
    const Mat sElement = structuringElement(openingKsize, cv::MORPH_RECT);
    Mat textEnhanced = textEnhance(whiteOnBlack, sElement, false);


    Mat vLines;
    Mat hLines;
    // detect large horizontal and vertical lines
    cv::morphologyEx(textEnhanced, hLines, cv::MorphTypes::MORPH_OPEN, structuringElement(250, 5, cv::MORPH_RECT));
    cv::morphologyEx(textEnhanced, vLines, cv::MorphTypes::MORPH_OPEN, structuringElement(5, 250, cv::MORPH_RECT));

    //Mat opened;
    //cv::morphologyEx(linesRemoved, opened, cv::MorphTypes::MORPH_OPEN, structuringElement(12, 12, cv::MORPH_ELLIPSE));

    // (gaussian) blur -> matches vision character
    // before doing the morphological operation - makes intensities more uniform in th
    // don't use the binarised (or blurred) image for OCR (don't throw away)
    // histogram of gradients

    //clean it up a bit?
    // shapes: MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    //cv::morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_ERODE, structuringElement(2, cv::MORPH_ELLIPSE));
    //cv::morphologyEx(preprocessed, preprocessed, cv::MorphTypes::MORPH_OPEN, structuringElement(1, 7, cv::MORPH_RECT));

    //int C = 0; // constant subtracted from calculated threshold value to obtain T(x, y)
    //cv::adaptiveThreshold(open, binarised, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 399, C);
    //cv::morphologyEx(binarised, binarised, cv::MorphTypes::MORPH_OPEN, structuringElement(2, cv::MORPH_ELLIPSE));
    //cv::threshold(hLines, hLinesBin, 0, 255, cv::THRESH_TOZERO | cv::THRESH_OTSU);
    //cv::threshold(vLines, vLinesBin, 0, 255, cv::THRESH_TOZERO | cv::THRESH_OTSU);

    // remove lines
    Mat preprocessed;
    cv::subtract(textEnhanced, hLines, preprocessed, cv::noArray(), CV_8U);
    cv::subtract(preprocessed, vLines, preprocessed, cv::noArray(), CV_8U);
    if (!batchMode) {
        showImage(grey8, "image");
        showImage(textEnhanced, "textEnhanced");
        //showImage(hLines, "hlines");
        //showImage(vLines, "vlines");
        showImage(preprocessed, "preprocessed");
    }

    return preprocessed;

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

    vector<CC> allCCs;

    for (int label = 0; label < nlabels; ++label) {
        auto left = stats.at<int>(label, cv::CC_STAT_LEFT);
        auto top = stats.at<int>(label, cv::CC_STAT_TOP);
        auto width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        auto height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        CC cc(left, top, width, height);
        cc.label = label;
        cc.area = stats.at<int>(label, cv::CC_STAT_AREA);
        cc.centroidX = centroids.at<double>(label, 0);
        cc.centroidY = centroids.at<double>(label, 1);
        allCCs.push_back(cc);
    }

    vector<meanShift::Point<CC>> yCentroids;
    for (const auto &cc: allCCs) {
        if (isPlausibleCCSize(cc, binarised.cols, binarised.rows)) {
            //yCentroids.push_back(meanShift::Point {i, {centroidY}});
            // include height and width in clustering decision
            //yCentroids.emplace_back(meanShift::Point<CC>{cc, {cc.centroidY, (double)cc.height}});
            yCentroids.emplace_back(meanShift::Point<CC>{cc, {cc.centroidY}});
        }
    }
    if (!batchMode) {
        Mat allComponents = binarised.clone();
        Mat allowedComponents = binarised.clone();
        for (const auto &cc: allCCs) {
            if (isPlausibleCCSize(cc, binarised.cols, binarised.rows)) {
                drawCC(allComponents, cc);
                drawCC(allowedComponents, cc);
            }
            else if (cc.area >= minCCArea(binarised.cols, binarised.rows)) {
                drawCC(allComponents, cc);
            }
        }
        showImage(allComponents, "all CCs");
        showImage(allowedComponents, "plausible CCs");
    }

    // TODO justify bandwidth parameter
    auto ccClusters = meanShift::cluster(yCentroids, rowClusterBandwidth(binarised.rows));
    {
        // sort by increasing mode (Y Coordinate)
        using ccCluster = meanShift::Cluster<CC>;
        std::sort(ccClusters.begin(), ccClusters.end(), [](const ccCluster& c1, const ccCluster& c2) -> bool {
            return c1.getMode()[0] < c2.getMode()[0];
        });
    }
    /*
    // show cluster modes
    for (Cluster& c : ccClusters) {
        int mode = static_cast<int>(c.mode[0]);
        cv::line(allComponents, cv::Point(0, mode), cv::Point(allComponents.cols, mode), 255, thickness=3);
    }
    */

    // BELOW:
    // Essentially find outliers of row height for each row, and remove the corresponding bounding boxes/connected components

    /*
     * Within each cluster above, cluster again based on bounding box height, and keep only the largest cluster
     */
    vector<ccCluster> clustersByCentroid;
    clustersByCentroid.reserve(ccClusters.size());
    vector<ccCluster> allClustersByCentroid;
    constexpr int MAX_ROW_CLUSTER_SIZE = 40;
    for (ccCluster &ithCluster : ccClusters) {
        const auto CCs = ithCluster.getData();
        vector<meanShift::Point<CC>> ccLabelsInCluster;
        ccLabelsInCluster.reserve(CCs.size());
        for (const CC& cc : CCs) {
            // TODO add width too?
            ccLabelsInCluster.emplace_back(meanShift::Point<CC>{cc, {(double)cc.height()}});
        }
        // TODO justify bandwidth parameter
        // TODO try making it the average height of each row... e.g image height divided by number of clusters by centroid
        auto heightClusters = meanShift::cluster<CC>(ccLabelsInCluster, heightClusterBandwidth(binarised.rows));
        using ccCluster = meanShift::Cluster<CC>;
        std::sort(heightClusters.begin(), heightClusters.end(), [](const ccCluster& c1, const ccCluster& c2) -> bool {
            return c1.getSize() > c2.getSize();
        });

        // save biggest, subject to not being too big
        bool foundLargestAcceptable = false;
        for (const auto& c : heightClusters) {
            allClustersByCentroid.push_back(c);
            if (c.getSize() <= MAX_ROW_CLUSTER_SIZE && !foundLargestAcceptable) {
                clustersByCentroid.push_back(c);
                foundLargestAcceptable = true;
            }
            //printf("Cluster size: %d\n", c.getSize());
        }
    }

    if (!batchMode) {
        showCentroidClusters(binarised, allClustersByCentroid, "all height clusters in row");
        showCentroidClusters(binarised, clustersByCentroid, "largest height cluster in row");
    }
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
    rows.reserve(clustersByCentroid.size());
    // TODO maybe just cluster by width and X centroid?
    // TODO or estimate columns first, based on the raw connected components
    for (const ccCluster &cluster: clustersByCentroid) {
        vector<wordBB> bbsInCluster;
        bbsInCluster.reserve(cluster.getSize());
        for (const CC& c : cluster.getData()) {
            bbsInCluster.push_back(c.getBox());
        }

        // TODO group if they are closer than 1.5* median character width
        // TODO prevent them from getting too big?
        // TODO also make sure that they overlap vertically

        auto combinedWords = wordBB::combineHorizontallyClose(bbsInCluster, 1.5, Interval::ExpandType::AVG);
        vector<wordBB> likelyWords;
        // make rects for each partition
        for (const wordBB& w : combinedWords) {
            // simple filtering: remove unlikely sized 'words'
            if (isPlausibleWordBBSize(w, binarised.cols, binarised.rows)) {
                likelyWords.push_back(w);
            }
        }
        rows.push_back(likelyWords);
    }
    return rows;
}

static int estimateNumberOfColumns(const vector<wordBB>& wordBBs, int width, const Mat& rects, bool batchMode) {
    // this will count how many rects (from eligible rows) would be intersected by a cut at the given X coordinate
    Mat rectsCutByXCoord(width, 1, CV_64FC1, cv::Scalar(0.0));
    for (const auto& word: wordBBs) {
        for (auto j = word.left(); j < word.right(); ++j) {
            rectsCutByXCoord.at<double>(j) += 1.0;
        }
    }

    Mat rectCutDensity;
    {
        // have to make blur size odd
        auto blurSize = width/8 + ((width/8) % 2 == 0);
        // treat 'outside the image' as zero
        cv::GaussianBlur(rectsCutByXCoord, rectCutDensity, cv::Size(blurSize, blurSize), 0, 0, cv::BORDER_ISOLATED);

#ifndef REFERENCE_ANDROID
        using cv::plot::Plot2d;
        if (!batchMode) {
            cv::Ptr<Plot2d> plotCounts = makePlot(rectsCutByXCoord, &rects);
            cv::Ptr<Plot2d> plotSmoothedCounts = makePlot(rectCutDensity, &rects);
            Mat plotResultCounts;
            Mat plotResultSmoothedCounts;
            plotCounts->render(plotResultCounts);
            plotSmoothedCounts->render(plotResultSmoothedCounts);
            showImage(0.5 * plotResultCounts + 0.5 * rects, "words cut by X coordinate");
            showImage(0.5 * plotResultSmoothedCounts + 0.5 * rects, "smoothed count data");
        }
#endif // REFERENCE_ANDROID
    }

    unsigned int estimatedColumns;

    // count number of peaks by sign counting, where the sign is taken relative to a threshold (3rd quartile?)
    // TODO findpeaks needs improvement
    {
        Mat m0;
        cv::sort(rectCutDensity, m0, cv::SORT_ASCENDING | cv::SORT_EVERY_COLUMN);
        int n = rectCutDensity.rows;
        // 75th percentile
        auto q75 = m0.at<double>(n * 3 / 4);
        // 60th percentile
        auto q60 = m0.at<double>(n * 3 / 5);
        // count peaks by seeing when the rectCutDensity crosses the threshold
        unsigned int peaks = 0;
        bool inPeak = false;
        for (int i = 0; i < n; ++i) {
            auto x = rectCutDensity.at<double>(i);
            if (x >= q75) {
                // it's a peak
                if (!inPeak) {
                    peaks++;
                }
                inPeak = true;
            } else if (x <= q60) {
                // it's not a peak
                inPeak = false;
            }
        }
        estimatedColumns = peaks;
        if (!batchMode) {
            printf("Estimated columns by sign counting: %u\n", estimatedColumns);
        }
#ifndef REFERENCE_ANDROID
        using cv::plot::Plot2d;
        if (!batchMode) {
            cv::Ptr<Plot2d> plotThreshold = makePlot(rectCutDensity, &rects);
            Mat plotResultThresh;
            {
                plotThreshold->render(plotResultThresh);
                // draw q3 as a line on the image
                // need to calculate its y coordinate, given that the plot's original height was equal to
                // max(smoothedCutRectDensity), but was rescaled to have height equal to the original image
                double maxVal;
                cv::minMaxIdx(rectCutDensity, nullptr, &maxVal);
                // subtract from 1.0 to get threshold referred to bottom of image, not top
                auto q75YCoord = static_cast<int>((1.0 - q75 / maxVal) * plotResultThresh.rows);
                auto q60YCoord = static_cast<int>((1.0 - q60 / maxVal) * plotResultThresh.rows);
                auto red = cv::Scalar(0, 0, 255); // B, G, R
                cv::rectangle(plotResultThresh, cv::Rect(0, q75YCoord, plotResultThresh.cols - 1, 1), red, 5);
                cv::rectangle(plotResultThresh, cv::Rect(0, q60YCoord, plotResultThresh.cols - 1, 1), red, 5);
            }
            showImage(0.5 * plotResultThresh + 0.5 * rects);
        }
#endif // REFERENCE_ANDROID
    }
    return estimatedColumns;
}

static void classifyColumns(vector<wordBB>& words, int numColumns, const Mat& rectsImg, bool batchMode) {
    // now put horizontal centres of all wordBBs into a Mat and run kmeans
    if (words.empty()) {
        fprintf(stderr, "classifyColumns(): warning: words list was empty");
        return;
    }
    Mat wordBBcentroidX((int) words.size(), 1, CV_64FC1);
    for (unsigned int i = 0; i < words.size(); ++i) {
        wordBBcentroidX.at<double>(i, 0) = words[i].boxCentreX();
    }

    if (!batchMode) {
        printf("Mixture model with %u components and centroid X values:\n", numColumns);
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
    emAlg->setClustersNumber(numColumns);
    //emAlg->setTermCriteria(termCriteria);
    // since we're in 1D, COV_MAT_DIAGONAL (separate sigma for each dim)
    // is equivalent to COV_MAT_SPHERICAL (all dims have same sigma)
    emAlg->setCovarianceMatrixType(cv::ml::EM::Types::COV_MAT_SPHERICAL);
    Mat initialMeans(numColumns, 1, CV_64FC1);
    for (auto i = 0; i < numColumns; ++i) {
        // want the centre (hence +0.5) of the ith column,
        // when image is divided into numColumns parts of equal width
        initialMeans.at<double>(i, 0) = (i + 0.5) / numColumns * rectsImg.cols;
    }
    if (!batchMode) {
        for (auto i = 0; i < numColumns; ++i) {
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
    Mat weights = emAlg->getWeights();
    std::vector<Mat> covs;
    covs.reserve(numColumns);
    emAlg->getCovs(covs);
    if (!batchMode) {
        for (auto k = 0; k < numColumns; ++k) {
            printf("Mixture %d: weight=%f mean=%f variance=%f\n", k, weights.at<double>(k), means.at<double>(k)
                   , covs[k].at<double>(0));
        }
#ifndef REFERENCE_ANDROID
        // plot density
        Mat density(rectsImg.cols, 1, CV_64FC1, 0.0);
        Mat samples(rectsImg.cols, 1, CV_64FC1);
        for (int i = 0; i < rectsImg.cols; ++i) {
            samples.at<double>(i) = (double) i;
        }
        /*
         * calculate density using vector operations
         * density(i) =
         *  sum_{k= 0 to numColumns - 1} of
         *      weights[k] * sqrt/(2*pi*variance[k] * exp(-0.5/variance[k] * (samples(i) - means[k])^2)
         */
        // 1/sqrt(2) * 2/sqrt(pi) / 2
        constexpr auto M_1_SQRT_2PI = M_SQRT1_2*M_2_SQRTPI/2.0;
        for (auto k = 0; k < numColumns; ++k) {
            auto variance = covs[k].at<double>(0);
            auto constant = weights.at<double>(k) * M_1_SQRT_2PI / sqrt(variance);
            Mat meansSubtracted = samples - means.at<double>(k);
            Mat scaled = meansSubtracted.mul(meansSubtracted, -0.5/variance);
            Mat exponentiated;
            cv::exp(scaled, exponentiated);
            density += constant*exponentiated;
        }

        // now plot it
        cv::Ptr<cv::plot::Plot2d> plotDensity = makePlot(density, &rectsImg, cv::Scalar(255, 255, 0));
        Mat plotResultDensity;
        plotDensity->render(plotResultDensity);
        showImage(0.5 * plotResultDensity + 0.5 * rectsImg, "Fitted mixture model for column distribution");
#endif // REFERENCE_ANDROID
    }

    /*
    //Mat outputCentres;
    cv::kmeans(wordBBcentroidX, numColumns, bestLabels, termCriteria, 3, cv::KMEANS_PP_CENTERS, outputCentres);
    */
    // apply column labels
    {
        /* Labels might not be in order of left-to-right columns, so we need to create this mapping */
        vector<int> labelOrder(numColumns, 0);
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

static void doOcr(const Mat& imageForOcr, tesseract::TessBaseAPI& tesseractAPI, vector<wordBB> & allWordBBs) {
    auto bytes_per_line = static_cast<int>(imageForOcr.step1() * imageForOcr.elemSize());
    tesseractAPI.SetImage(imageForOcr.data, imageForOcr.cols, imageForOcr.rows, /*bytes_per_pixel=*/1, bytes_per_line);
    tesseractAPI.SetSourceResolution(300);

    for (wordBB &w : allWordBBs) {
        /*
        Mat tessImage;
        w.text = getCleanedText(tesseractAPI, w, tessImage);
        showImage(tessImage, "OCR for wordBB");
        /*/
        w.setText(getCleanedText(tesseractAPI, w));
        /**/
    }
}

