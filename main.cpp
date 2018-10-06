#include "reference.h"
#include "utils.h"
#include "ocrutils.h"
#include "InputParser.h"

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

std::string baseName(const char * fileName) {
    std::string basename(fileName);
    size_t lastSlash = basename.rfind('/');
    if (lastSlash != std::string::npos) {
        basename.erase(0, lastSlash);
    }
    return basename;
}

void testMain() {
    std::string filePath("est-images/output//home/max/uni/thesis/label-pics-cropped/img_2557.txt");
    std::string fileString = readFile(filePath);
    printf("Read string: %s\n", fileString.c_str());
    Table t = Table::parseFromString(fileString, "\\");
    std::cout << "Table:" << t.printableString(30) << std::endl;
}

static const char configPath[] = "/home/max/thesis/koala/data/tesseract.config";
static const char dataPath[] = "/usr/share/tessdata/";

static std::pair<double, double> doTableComparison(const Table& test, const string& truthFile, const string& testOutPath) {
    bool doOutput = !testOutPath.empty();
    string trueTableString = readFile(truthFile);
    if (trueTableString.empty()) {
        fprintf(stderr, "Ground truth table string could not be read");
        return {-1, -1};
    }
    Table trueTable = Table::parseFromString(trueTableString, "\\");
    std::pair<double, double> comparisonScore = Table::compareTable(test, trueTable);

    if (doOutput) {
        std::ofstream testOutFile(testOutPath);
        using std::endl;
        if (testOutFile.is_open()) {
            testOutFile << endl << endl;
            testOutFile << "**** Ground truth table comparison: ****" << endl;
            testOutFile << "Ground truth table:" << endl;
            testOutFile << trueTable.printableString(25) << endl;
            testOutFile << endl << endl;
            testOutFile << "Actual table:" << endl;
            testOutFile << test.printableString(25) << endl;
            testOutFile << endl << endl;
            testOutFile << "Comparison scores:" << endl;
            testOutFile << "Average key column accuracy: " << 100*comparisonScore.first << "%" << endl;
            testOutFile << "Weighted value col accuracy: " << 100*comparisonScore.second << "%" << endl;
        } else {
            fprintf(stderr, "Could not write to test output file");
        }
    }

    return comparisonScore;
}

int main(int argc, char ** argv) {
    // TODO add option to suppress image display
    if (argc != 3 && argc != 4) {
        printf("Usage: %s <input.img> [-o <output prefix>] [-t <ground truth.txt>]\n", argv[0]);
        return -1;
    }

    InputParser ip(argc, argv);
    if (ip.cmdOptionExists("-o")) {

    }

    const char * inFile = argv[1];
    std::string outPrefix = ip.getCmdOption("-o");
    std::string truthFile = ip.getCmdOption("-t");

    bool doTest = !truthFile.empty();
    bool doOutput = !outPrefix.empty();
    // if we have a ground truth file, assume it's batch mode
    bool batchMode = doTest;

    cv::Mat image = imread(inFile, cv::IMREAD_GRAYSCALE);
    if (!image.data) {
        fprintf(stderr, "Could not read input image!\n");
        return 1;
    }

    tesseract::TessBaseAPI tesseractAPI;
    if (tesseractInit(tesseractAPI, dataPath, configPath) == -1) {
        fprintf(stderr, "Could not initialise tesseract API");
        return 1;
    }

    Mat clusteredWords;
    Table outTable = tableExtract(image, tesseractAPI, &clusteredWords, batchMode);
    if (!batchMode) {
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

    if (doTest) {
        std::string testOutputPath = doOutput ? outPrefix.append(".test") : "";
        auto scores = doTableComparison(outTable, truthFile, testOutputPath);
        printf("%s %.3f %.3f\n", baseName(inFile).c_str(), scores.first, scores.second);
    } else {
        std::cout << outTable.printableString(30);
    }
}
