//
// Created by max on 10/7/18.
//

#ifndef REFERENCE_TABLECOMPARISON_H
#define REFERENCE_TABLECOMPARISON_H

#include "table.h"

#include <string>

struct comparisonResult {
    double keyScore;
    double valScore;
    int expectedCols;
    int actualCols;

};


comparisonResult doTableComparison(const Table& test, const std::string& truthFile, const std::string& testOutPath);

#endif //REFERENCE_TABLECOMPARISON_H
