#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cstdio>
#include <string>
#include <algorithm>

// android build uses different header files
#ifdef REFERENCE_ANDROID
#include <baseapi.h>
#else
#include <tesseract/baseapi.h>
#include <fstream>

#endif

#include "reference.h"
#include "utils.h"
#include "ocrutils.h"

/*
std::string exec(const char * cmd) {
    constexpr int bufferLength = 128;
    char buffer[bufferLength];

    FILE * pipe = popen(cmd, "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    std::string result = "";
    try {
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL)
                result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

std::string baseName(const char * fileName) {
    std::string basename(fileName);
    size_t lastSlash = basename.rfind('/');
    basename.erase(0, lastSlash);
    return basename;
}
*/

int main(int argc, char ** argv) {
    if (argc != 3) {
        printf("Usage: %s <input.img> <output prefix>\n", argv[0]);
        return -1;
    }

    const char * inFile = argv[1];
    const char * outFile = argv[2];

    std::string outCsv = outFile;
    outCsv.append(".csv");


    cv::Mat image = imread(inFile, cv::IMREAD_GRAYSCALE);
    if (!image.data) {
        fprintf(stderr, "Could not read input image!\n");
        return 1;
    }

    char configPath[] = "/home/max/thesis/koala/data/tesseract.config";
    char dataPath[] = "/usr/share/tessdata/";
    tesseract::TessBaseAPI tesseractAPI;
    if (tesseractInit(tesseractAPI, dataPath, configPath) == -1) {
        fprintf(stderr, "Could not initialise tesseract API");
        return 1;
    }

    Mat clusteredWords;
    Table outTable = tableExtract(image, tesseractAPI, &clusteredWords);
    showImage(clusteredWords);

    tesseractAPI.End();

    std::cout << outTable.printableString(30);

    std::ofstream tableOutput(outCsv, std::ios::binary);
    std::string outString = outTable.parseableString(",");
    // make sure characters are ascii!!
    // ::tolower uses the 'tolower' function in the outermost namespace
    std::transform(outString.begin(), outString.end(), outString.begin(), ::tolower);
    tableOutput << outString;

}
