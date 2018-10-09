#include "reference.h"
#include "helpers.h"
#include "ocrutils.h"
#include "InputParser.h"
#include "tableComparison.h"

#include <iostream>
#include <cstdio>
#include <string>
#include <fstream>

// android build uses different header files
#ifdef REFERENCE_ANDROID
#include <baseapi.h>
#else
#include <tesseract/baseapi.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using std::string;
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
*/

void testMain() {
    std::string filePath("est-images/output//home/max/uni/thesis/label-pics-cropped/img_2557.txt");
    std::string fileString = readFile(filePath);
    printf("Read string: %s\n", fileString.c_str());
    Table t = Table::parseFromString(fileString, "\\");
    std::cout << "Table:" << t.printableString(30) << std::endl;
}

static const char configPath[] = "/home/max/thesis/koala/data/tesseract.config";
static const char dataPath[] = "/usr/share/tessdata/";


int main(int argc, char ** argv) {
    // TODO add option to suppress image display
    if (argc < 2 || argc > 7) {
        printf("Usage: %s <input.img> [-o <output prefix>] [-t <ground truth.txt>] [-v]\n", argv[0]);
        printf("Test output format: <filename>: <key col score> <value col score> <est. columns> <column discrepancy>\n");
        return -1;
    }

    InputParser ip(argc, argv);
    string inFile = ip.getArg(0);
    string outPrefix = ip.getCmdOption("-o");
    string truthFile = ip.getCmdOption("-t");

    bool doTest = !truthFile.empty();
    bool doOutput = !outPrefix.empty();
    // if we have a ground truth file, assume it's batch mode
    bool batchMode = doTest && !ip.cmdOptionExists("-v");


    cv::Mat image = imread(inFile, cv::IMREAD_GRAYSCALE);
    if (!image.data) {
        fprintf(stderr, "Could not read input image '%s'!\n", inFile.c_str());
        return 1;
    }

    tesseract::TessBaseAPI tesseractAPI;
    if (tesseractInit(tesseractAPI, dataPath, configPath) == -1) {
        fprintf(stderr, "Could not initialise tesseract API");
        return 1;
    }

    cv::Mat clusteredWords;
    Table outTable = tableExtract(image, tesseractAPI, &clusteredWords, batchMode);
    if (!batchMode && !clusteredWords.empty()) {
        showImage(clusteredWords);
    }

    tesseractAPI.End();


    if (doOutput) {
        std::string tableOutputPath = std::string(outPrefix).append(".table");
        std::ofstream tableOutput(tableOutputPath, std::ios::binary);
        // make sure characters are ascii!!
        // ::tolower uses 'tolower' function from outermost namespace
        // std::transform(outString.begin(), outString.end(), outString.begin(), ::tolower);
        if (tableOutput.is_open()) {
            tableOutput << outTable.parseableString(",");
            tableOutput.close();
        } else {
            fprintf(stderr, "could not open output file for csv writing: %s\n", tableOutputPath.c_str());
        }
    }
    if (!batchMode) {
        std::cout << outTable.printableString(30);
    }

    if (doTest) {
        std::string testOutputPath = doOutput ? outPrefix.append(".test") : "";
        auto name = basename(inFile).c_str();
        auto s = doTableComparison(outTable, truthFile, testOutputPath);
        printf("%s %.3f %.3f %+d\n", name, s.keyScore, s.valScore, s.actualCols - s.expectedCols);
    }

}
