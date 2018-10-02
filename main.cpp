#include "reference.h"
#include "utils.h"
#include "ocrutils.h"

#include <iostream>
#include <cstdio>
#include <string>
#include <algorithm>
#include <fstream>

// android build uses different header files
#ifdef REFERENCE_ANDROID
#include <baseapi.h>
#else
#include <tesseract/baseapi.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

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

void testMain() {
    std::string filePath("est-images/output//home/max/uni/thesis/label-pics-cropped/img_2557.txt");
    std::string fileString = readFile(filePath);
    printf("Read string: %s\n", fileString.c_str());
    Table t = Table::parseFromString(fileString, "\\");
    std::cout << "Table:" << t.printableString(30) << std::endl;
}

static const char configPath[] = "/home/max/thesis/koala/data/tesseract.config";
static const char dataPath[] = "/usr/share/tessdata/";

static void doTableComparison(const Table& test, const char * truthFile) {
    std::string trueTableString = readFile(truthFile);
    if (trueTableString == "") {
        printf("Ground truth table string could not be read");
        return;
    }
    Table trueTable = Table::parseFromString(trueTableString, "\\");

    using namespace std;
    pair<double, double> comparisonScore = Table::compareTable(test, trueTable);
    cout << endl << endl;
    cout << "**** Ground truth table comparison: ****" << endl;
    cout << "Ground truth table:" << endl;
    cout << trueTable.printableString(25) << endl;
    cout << "Comparison scores:" << endl;

    printf("Avg Levenshtein distance score (key column): %.1f%%\n", 100*comparisonScore.first);
    printf("Weighted avg Levenshtein score (value cols): %.1f%%\n", 100*comparisonScore.second);
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        printf("Usage: %s <input.img> <output prefix> [<ground truth.txt>]\n", argv[0]);
        return -1;
    }

    const char * inFile = argv[1];
    const char * outFile = argv[2];
    const char * truthFile = argc == 4 ? argv[3] : nullptr;

    std::string outCsv = outFile;
    outCsv.append(".csv");

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
    Table outTable = tableExtract(image, tesseractAPI, &clusteredWords);
    showImage(clusteredWords);

    tesseractAPI.End();

    std::cout << outTable.printableString(30);

    std::ofstream tableOutput(outCsv, std::ios::binary);
    std::string outString = outTable.parseableString(",");
    // make sure characters are ascii!!
    // ::tolower uses the 'tolower' function in the outermost namespace
    // std::transform(outString.begin(), outString.end(), outString.begin(), ::tolower);
    tableOutput << outString;

    if (truthFile != nullptr) {
        doTableComparison(outTable, truthFile);
    }
}
