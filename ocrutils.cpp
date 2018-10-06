//
// Created by max on 9/23/18.
//

#include <regex>

#include "ocrutils.h"

// returns 8-bit single channel Mat from corresponding binary pix image
cv::Mat matFromPix1(PIX * pix) {
    cv::Mat mat(pix->h, pix->w, CV_8UC1);
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


// config file settings:
// load_system_dawg 0
// (don't load system dictionary)
// TODO load user dictionary
int tesseractInit(tesseract::TessBaseAPI& baseAPI, const char * path, const char * configPath) {
    // need this for C++ reasons
    char * configPtr = const_cast<char *>(configPath);
    int tessStatus = baseAPI.Init(path, "eng", tesseract::OcrEngineMode::OEM_TESSERACT_ONLY, &configPtr, 1, nullptr, nullptr, true);
    //int tessStatus = baseAPI.Init(path, "eng", tesseract::OcrEngineMode::OEM_DEFAULT, &configPtr, 1, nullptr, nullptr, true);
    //int tessStatus = tesseractAPI.Init("/usr/share/tessdata/", "eng", tesseract::OcrEngineMode::OEM_LSTM_ONLY);
    if (tessStatus == -1) {
        return tessStatus;
    }
    //tesseractAPI.ReadConfigFile();

    // tesseractAPI.SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK_VERT_TEXT);
    baseAPI.SetPageSegMode(tesseract::PageSegMode::PSM_RAW_LINE);

    // NOTE this doesn't work when using the LSTM functionality of tesseract
    const char * whitelistChars = "1234567890%(),-.<>abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const char * blacklistChars = "{}|$#@!";
    baseAPI.SetVariable("tessedit_char_whitelist", whitelistChars);
    baseAPI.SetVariable("tessedit_char_blacklist", blacklistChars);
    //cv::Ptr<cvTess> ocr = cvTess::create("/usr/share/", "eng", whitelistChars, cv::text::PSM_AUTO);
    return 0;

}


// API must be initialised with image
std::string getCleanedText(tesseract::TessBaseAPI &tesseractAPI, const wordBB& w) {
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

    tesseractAPI.SetRectangle(w.x, w.y, w.width, w.height);

    // assume only ASCII characters, by using whitelist
    const char *out = tesseractAPI.GetUTF8Text();
    std::string ocrText = out; // copy constructor
    delete[] out;

    // remove newlines
    ocrText.erase(remove(ocrText.begin(), ocrText.end(), '\n'), ocrText.end());
    // erase repeated spaces
    std::regex repeatedSpace("\\s\\s*");
    std::string despaced = std::regex_replace(ocrText, repeatedSpace, " ");

    return despaced;
}

std::string getCleanedText(tesseract::TessBaseAPI &tesseractAPI, const wordBB& w, cv::Mat& tessImage) {
    std::string ocrText = getCleanedText(tesseractAPI, w);

    Pix * img = tesseractAPI.GetThresholdedImage();
    // create mat from img data. step parameter is number of bytes per row, wpl is (integer) words per line
    //PIX * converted = pixConvert1To32(nullptr, img, 0, 255);
    //Mat tessPix(converted->h, converted->w, CV_8UC4, converted->data, sizeof(int)*converted->wpl);
    tessImage = matFromPix1(img);
    pixDestroy(&img);

    return ocrText;
}
