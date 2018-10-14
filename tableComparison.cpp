//
// Created by max on 10/7/18.
//

#include "tableComparison.h"
#include "table.h"
#include "helpers.h"
#include "levenshtein.h"

#include <vector>
#include <fstream>

using std::string;

template<typename T>

// helper method to do cast from int to size_t
static void reserveSpace(std::vector<T>& v, int space) {
    v.reserve(static_cast<size_t>(space));
}

/*
 * Table comparison algorithm.
 * Terminology:
 *     'actual' table is the one produced by the algorithm. 'Expected' is the ground truth
 *     'key' column is the first column, other columns are 'value' columns
 *     (this is an assumption on the structure of nutrition tables)
 */
comparisonResult compareTable(const Table& actual, const Table& expected) {
    if (actual.numCols() == 0 || expected.numCols() == 0) {
        // full marks if they match, zero if they don't
        double score = actual.numCols() == expected.numCols() ? 1.0 : 0.0;
        return {score, score};
    }
    // from now on assume that both tables have at least one column

    // 1. Use the first column to match/assign rows of 'actual' to 'expected':
    int A = actual.numRows();
    int E = expected.numRows();

    // holds row matchings (actual-expected row pairings, **in that order**)
    // these will be sorted by comparing corresponding entries in distances (distances[i][j])
    using std::vector;
    using std::pair;
    vector<pair<int, int>> indexPairs;
    vector<vector<double>> distances;
    reserveSpace(indexPairs, A*E);
    reserveSpace(distances, A);
    {
        for (decltype(A) i = 0; i < A; ++i) {
            vector<double> ithRowDistances;
            reserveSpace(ithRowDistances, E);
            for (decltype(E) j = 0; j < E; ++j) {
                indexPairs.emplace_back(pair<int, int>(i, j));
                ithRowDistances.push_back(asymStringSimilarity(actual.getText(i, 0), expected.getText(j, 0)));
            }
            distances.push_back(ithRowDistances);
        }
        // sort scores in descending order
        sort(indexPairs.begin(), indexPairs.end(), [&distances](pair<int, int> a, pair<int, int> b) -> bool {
            return distances[a.first][a.second] > distances[b.first][b.second];
        });
    }

    // mapping from rows in Expected to assigned rows in Actual (injective mapping, not necessarily surjective)
    vector<int> actualRowForExpRow(E, 0);
    {
        vector<bool> expectedRowAssigned((size_t)E, false);
        vector<bool> actualRowAssigned((size_t)A, false);
        decltype(E) rowsAssigned = 0;
        // now the pairs are sorted with closest string pairs first
        // just greedily assign the expected table rows to the actual rows, by closest string matching
        for (const pair<int, int> &matchings : indexPairs) {
            auto actualRowIdx = matchings.first;
            auto expectedRowIdx = matchings.second;
            if (!expectedRowAssigned[expectedRowIdx] && !actualRowAssigned[actualRowIdx]) {
                // assign it
                actualRowForExpRow[expectedRowIdx] = actualRowIdx;
                expectedRowAssigned[expectedRowIdx] = true;
                actualRowAssigned[actualRowIdx] = true;
                rowsAssigned++;
                if (rowsAssigned == E) {
                    // finished assigning all rows in ground truth table
                    break;
                }
            }
        }
    }

    // now go through each matching and add up the levenshtein distance for the whole row
    /*
    if (expected.numCols() != actual.numCols()) {
        printf("Warning: expected table has %d cols but actual table has %d cols\n", expected.numCols(), actual.numCols());
    }
    */

    // levenshtein distance
    vector<double> keyColScores; // holds levenshtein distance for the between first columns (keys) in each row
    vector<double> avgValueColScores; // holds the average levenshtein distance for the remaining columns in each row
    {
        reserveSpace(keyColScores, E);
        reserveSpace(avgValueColScores, E);
        for (decltype(E) expectedRowIdx = 0; expectedRowIdx < E; ++expectedRowIdx) {
            const auto& actualRow = actual.getRow(actualRowForExpRow[expectedRowIdx]);
            const auto& expectedRow = expected.getRow(expectedRowIdx);
            keyColScores.push_back(asymStringSimilarity(actualRow[0], expectedRow[0]));

            // find the average levenshtein distance, assuming that the value columns match up
            // if the actual and expected tables have a different number of columns,
            // pretend that the table with fewer columns has just empty strings in those extra columns

            auto numValueColumns = std::max(actual.numCols(), expected.numCols()) - 1; // no underflow due to earlier check
            /* Compare as many columns as we can: from 1 to min(actual.columns, expected.columns) - 1
             * The remaining columns implicitly get a score of 0, because we use
             *    numValueColumns = max(actual.columns, expected.columns) - 1 (which is > 0)
             * as the normalising constant
             */
            auto numValueColumnsForCompare = std::min(actual.numCols(), expected.numCols()) - 1;
            double avgValueColScore;
            if (numValueColumns == 0) {
                avgValueColScore = 1.0; // nothing to compare, so perfect score
            } else {
                avgValueColScore = 0.0;
                for (decltype(numValueColumns) j = 0; j < numValueColumnsForCompare; ++j) {
                    avgValueColScore += asymStringSimilarity(actualRow[j+1], expectedRow[j+1]) / numValueColumns;
                }
            }
            avgValueColScores.push_back(avgValueColScore);
        }
    }


    const double sumKeyColScores = std::accumulate(keyColScores.begin(), keyColScores.end(), 0.0);
    // average levenshtein distance score in key column
    double avgKeyColScore = sumKeyColScores / E;
    // sum of levenshtein distance scores in value column;
    // weighted according to how well the corresponding key column matches the ground truth
    // the point of doing this is to try to minimise 'doubly penalising' badly matched key columns
    double weightedAvgValueColumnScore = 0.0;

    for (decltype(E) i = 0; i < E; ++i) {
        // if key col scores were zero match, just use equal weighting
        auto rowWeight = sumKeyColScores != 0 ? keyColScores[i] / sumKeyColScores : 1.0/E;
        weightedAvgValueColumnScore += rowWeight * avgValueColScores[i];
    }

    return {avgKeyColScore, weightedAvgValueColumnScore, expected.numCols(), actual.numCols()};
}

comparisonResult doTableComparison(const Table& test, const string& truthFile, const string& testOutPath) {
    bool doOutput = !testOutPath.empty();
    string trueTableString = readFile(truthFile);
    if (trueTableString.empty()) {
        fprintf(stderr, "Ground truth table string could not be read");
        return {-1, -1, 0, 0};
    }
    Table trueTable = Table::parseFromString(trueTableString, "\\");
    auto result = compareTable(test, trueTable);

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
            testOutFile << "Average key column accuracy: " << 100*result.keyScore << "%" << endl;
            testOutFile << "Weighted value col accuracy: " << 100*result.valScore << "%" << endl;
            testOutFile << "Expected number of columns:  " << result.expectedCols << endl;
            testOutFile << "Actual number of columns:  " << result.actualCols << endl;
        } else {
            fprintf(stderr, "Could not write to test output file");
        }
    }

    return result;
}
