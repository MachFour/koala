//
// Created by max on 10/7/18.
//

#include "tableComparison.h"
#include "table.h"
#include "helpers.h"
#include "levenshtein.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <fstream>

using std::vector;
// pair of row indices; used for matchings of actual and expected table rows
using idxPair = std::pair<size_t, size_t>;

comparisonResult compareTable(const Table& actual, const Table& expected) {
    const auto rowsA = actual.rows();
    const auto rowsE = expected.rows();
    const auto colsA = actual.cols();
    const auto colsE = expected.cols();
    const auto colDiff = Table::columnDifference(actual, expected);
    const auto s0 = decltype(rowsE){0}; // simpler loop bodies

    if (colsA == 0 || colsE == 0) {
        // full marks if they match, zero if they don't
        auto score = (colDiff == 0) ? 1.0 : 0.0;
        return {score, score, colDiff};
    }
    // from now on assume that both tables have at least one column

    // 1. Use key column to match/assign rows of 'actual' to 'expected'

    /* matchScore[i][j] is the string similarity between key i of
     * actual table (ith row, first column) and key j of expected table.
     *
     * matchings holds all possible pairs of row matchings (act : exp)
     * these will be sorted in order of decreasing match scores
     */
    vector<vector<double>> matchScore;
    vector<idxPair> matchings;
    auto sortOrder = [&matchScore]
            (const idxPair& a, const idxPair& b) -> bool {
        return matchScore.at(a.first).at(a.second) > matchScore.at(b.first).at(b.second);
    };

    {
        matchScore.reserve(rowsA);
        matchings.reserve(rowsA*rowsE);
        for (auto i = s0; i < rowsA; ++i) {
            vector<double> scoresForKeyI(rowsE);
            for (auto j = s0; j < rowsE; ++j) {
                auto strActual = actual.textAt(i, 0);
                auto strExpected = expected.textAt(j, 0);
                scoresForKeyI.at(i) = stringSimilarity(strActual, strExpected);
                matchings.emplace_back(idxPair{i, j});
            }
            matchScore.push_back(scoresForKeyI);
        }
        // sort scores by descending order of score
        sort(matchings.begin(), matchings.end(), sortOrder);
    }

    // Row correspondence: expected -> actual. Injective, not surjective
    vector<size_t> correspondingRow(rowsE, 0);
    {
        vector<bool> isRowAssignedE(rowsE, false);
        vector<bool> isRowAssignedA(rowsA, false);
        size_t rowsAssigned = 0;
        // just greedily assign rows, since matchings are sorted
        // in decreasing order of string similarity
        for (const auto& m : matchings) {
            auto rowA = m.first;
            auto rowE = m.second;
            if (!isRowAssignedE.at(rowE) && !isRowAssignedA.at(rowA)) {
                isRowAssignedE.at(rowE) = true;
                isRowAssignedA.at(rowA) = true;
                correspondingRow.at(rowE) = rowA;
                if (++rowsAssigned == rowsE) {
                    break; // finished assigning rows of expected table
                }
            }
        }
    }

    // 2. For each row in expected table, calculate string similarities
    // with key and value columns in the corresponding row of actual.

    /* For the key column, this is easy: just reuse the string similarity
     * found before. For the value columns, it's probably enough to find
     * find the average string similarity across columns, assuming that
     * they can be paired up sequentially. If actual and expected tables
     * differ in number of columns, compare as many as possible and
     * assign a score of zero for those remaining.
     * This is equivalent to comparing with empty strings.
     */

    vector<double> keyScore(rowsE, 0);
    vector<double> valueScore(rowsE, 0);
    {
        // no underflow due to earlier check
        auto numColumnsToCompare = std::min(colsA, colsE) - 1;
        // Normalising by larger number of columns does what we want here
        auto columnsToDivideBy = std::max(colsA, colsE) - 1;
        for (auto r = s0; r < rowsE; ++r) {
            const auto& cellsA = actual.getRow(correspondingRow.at(r));
            const auto& cellsE = expected.getRow(r);

            keyScore.at(r) = matchScore.at(correspondingRow.at(r)).at(r);
            if (columnsToDivideBy != 0) {
                // off by 1 because the key column is the first one
                for (auto j = s0 + 1; j < numColumnsToCompare + 1; ++j) {
                    auto score = stringSimilarity(cellsA.at(j), cellsE.at(j));
                    valueScore.at(r) += score / columnsToDivideBy;
                }
            } else {
                valueScore.at(r) = 1.0; // both tables have no value cols
            }
        }
    }

    auto sumKeyScore = std::accumulate(keyScore.begin(), keyScore.end(), 0.0);
    auto meanKeyScore = sumKeyScore / rowsE;
    /* Sum of string similarity score averaged across all value columns,
     * and weighted according to how similar the corresponding keys are.
     * Goal: try to minimise doubly penalising mismatched key columns.
     */

    auto weightedValueScore = 0.0;
    if (sumKeyScore == 0) {
        // just use equal weighting
        for (auto r = s0; r < rowsE; ++r) {
            weightedValueScore += valueScore.at(r) / rowsE;
        }
    } else {
        for (auto r = s0; r < rowsE; ++r) {
            weightedValueScore += valueScore.at(r) * keyScore.at(r)/sumKeyScore;
        }
    }

    return {meanKeyScore, weightedValueScore, colDiff};
}

comparisonResult doTableComparison(const Table& test, const std::string& truthFile, const std::string& testOutPath) {
    bool doOutput = !testOutPath.empty();
    std::string trueTableString = readFile(truthFile);
    if (trueTableString.empty()) {
        fprintf(stderr, "Ground truth table string could not be read");
        return {-1, -1, 0};
    }
    Table trueTable = Table::parseFromString(trueTableString, "\\");
    auto result = compareTable(test, trueTable);

    if (doOutput) {
        std::ofstream testOutFile(testOutPath);
        using std::endl;
        if (testOutFile.is_open()) {
            testOutFile << endl;
            testOutFile << "**** Ground truth table comparison: ****" << endl;
            testOutFile << "Ground truth table:" << endl;
            testOutFile << trueTable.printableString(25) << endl;
            testOutFile << endl << endl;
            testOutFile << "Actual table:" << endl;
            testOutFile << test.printableString(25) << endl;
            testOutFile << endl << endl;
            testOutFile << "Comparison scores:" << endl;
            testOutFile << "Average key column accuracy: " << 100*result.keyScore << "%" << endl;
            testOutFile << "Weighted value col accuracy: " << 100*result.valScore << "%" << endl;
            testOutFile << "Column estimation difference (actual - expected): " << result.colDiff << endl;
            testOutFile.close();
        } else {
            fprintf(stderr, "Could not write to test output file");
        }
    }

    return result;
}
