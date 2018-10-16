//
// Created by max on 10/7/18.
//

#ifndef REFERENCE_TABLECOMPARISON_H
#define REFERENCE_TABLECOMPARISON_H

#include "table.h"
#include <string>
/*
 * Table comparison algorithm.
 * Terminology:
 *   'actual' table is the one produced by the algorithm,
 *   'Expected' is the ground truth.
 *   'key' column is the first column, other columns are 'value' columns
 *   (this is a mild assumption on the structure of the tables)
 * Return value:
 *   keyScore: mean string similarity of corresponding cells in key column
 *      The correspondence between rows is chosen to maximise this score.
 *   valScore: weighted mean string similarity of corresponding value cells
 *      weight is proportional to how well their keys match.
 *   colDiff: difference in number of columns (actual - expected)
 */
struct comparisonResult {
    double keyScore;
    double valScore;
    int colDiff;
};

comparisonResult doTableComparison(const Table& test, const std::string& truthFile, const std::string& testOutPath);

#endif //REFERENCE_TABLECOMPARISON_H
