//
// Created by max on 10/1/18.
//

#include "table.h"
#include "levenshtein.h"

#include <vector>
#include <string>
#include <iterator>
#include <numeric>
#include <algorithm>

using string = std::string;
using stringVector = std::vector<std::string>;

/* Utility functions for this file */
static unsigned int rectifiedDifference(int a, int b);
static stringVector split(const string& test, const string& delim);

template <typename T>
static auto max(T x, T y) -> T;

// adapted from JDługosz's answer at
// https://codereview.stackexchange.com/questions/193203/splitting-stdstring-based-on-delimiter-using-only-find-and-substr



// column separators become '\f' characters
std::string Table::parseableString(const char * colSep) const {
    // takes into account padding characters
    std::string outStr;
    for (const stringVector& column : rows) {
        for (const std::string& cell : column) {
            outStr.append(cell);
            outStr.append(colSep);
        }
        // replace last column separator with newline to mark end of row
        outStr.back() = '\n';
    }
    return outStr;
}

string Table::printableString(unsigned int minColumnWidth) const {
    string outStr("\n\n");
    // takes into account padding characters
    int realMinColWidth = minColumnWidth - 2;
    for (const stringVector& column : rows) {
        for (const string& cell : column) {
            outStr.append(cell);
            auto fillChars = rectifiedDifference(realMinColWidth, (int) cell.length());
            outStr.append(fillChars, ' ');
            outStr.append("| ");
        }
        outStr.append("\n");
    }
    return outStr;
}

size_t Table::numRows() const {
    return rows.size();
}

// returns empty string if indices are out of range
string Table::getText(size_t row, size_t col) const {
    if (row >= numRows()) {
        throw std::invalid_argument("row index out of range");
    } else if (col >= columns) {
        throw std::invalid_argument("column index out of range");
    } else {
        return rows[row][col];
    }
}

const stringVector& Table::getRow(size_t row) const {
    if (row >= numRows()) {
        throw std::invalid_argument("row index out of range");
    } else {
        return rows[row];
    }
}

void Table::addRow() {
    stringVector row(columns);
    rows.push_back(row);
}
void Table::setColumnText(size_t row, size_t col, std::string text) {
    while (row >= rows.size()) {
        addRow();
    }
    rows[row][col] = text;
}

Table Table::parseFromString(string tableString, string columnSep) {
    stringVector rowStrings = split(tableString, "\n");
    std::vector<stringVector> columnSplits;
    columnSplits.reserve(rowStrings.size());

    size_t maxColumns = 0;
    for (string rowString : rowStrings) {
        if (rowString.find(columnSep) == std::string::npos) {
            // no column separators so it's probably garbage; ignore it
            continue;
        }
        stringVector cells = split(rowString, columnSep);
        maxColumns = max(maxColumns, cells.size());
        columnSplits.push_back(cells);
    }

    Table t(maxColumns);
    t.rows = columnSplits;
    return t;
}

static unsigned int rectifiedDifference(int a, int b) {
    return a - b <= 0 ? 0 : (unsigned int)(a - b);
}

template <typename T>
static auto max(T x, T y) -> T {
    return x > y ? x : y;
}

// adapted from JDługosz's answer at
// https://codereview.stackexchange.com/questions/193203/splitting-stdstring-based-on-delimiter-using-only-find-and-substr

static stringVector split(const string& test, const string& delim) {
    auto nextChar = test.cbegin();
    const auto stringEnd = test.cend();
    const auto delimStart = delim.cbegin();
    const auto delimEnd = delim.cend();
    const auto delimLength = delim.length();

    stringVector splitStrings;

    for (;;) {
        auto currentTokenEnd = std::search(nextChar, stringEnd, delimStart, delimEnd);
        splitStrings.emplace_back(nextChar, currentTokenEnd);
        if (currentTokenEnd == stringEnd) {
            break;
        }
        nextChar = currentTokenEnd + delimLength;
    }

    splitStrings.shrink_to_fit();
    return splitStrings;
}

/*
 * Table comparison algorithm.
 * Terminology:
 *     'actual' table is the one produced by the algorithm. 'Expected' is the ground truth
 *     'key' column is the first column, other columns are 'value' columns
 *     (this is an assumption on the structure of nutrition tables)
 */
auto Table::compareTable(const Table& actual, const Table& expected) -> std::pair<double, double> {
    if (actual.columns == 0 || expected.columns == 0) {
        // full marks if they match, zero if they don't
        double score = actual.columns == expected.columns ? 1.0 : 0.0;
        return {score, score};
    }
    // from now on assume that both tables have at least one column

    // 1. Use the first column to match/assign rows of 'actual' to 'expected':
    auto A = actual.numRows();
    auto E = expected.numRows();

    using namespace std;
    // holds row matchings (actual-expected row pairings, **in that order**)
    // these will be sorted by comparing corresponding entries in distances (distances[i][j])
    vector<pair<size_t, size_t>> indexPairs;
    indexPairs.reserve(A*E);
    {
        vector<vector<size_t>> distances;
        distances.reserve(A);
        for (decltype(A) i = 0; i < A; ++i) {
            vector<size_t> ithRowDistances;
            ithRowDistances.reserve(E);
            for (decltype(E) j = 0; j < E; ++j) {
                indexPairs.push_back({i, j});
                ithRowDistances.push_back(levenshtein(actual.getText(i, 0), expected.getText(j, 0)));
            }
            distances.push_back(ithRowDistances);
        }
        sort(indexPairs.begin(), indexPairs.end(), [&distances](pair<size_t, size_t> a, pair<size_t, size_t> b) -> bool {
            return distances[a.first][a.second] < distances[b.first][b.second];
        });
    }

    // mapping from rows in Expected to assigned rows in Actual (injective mapping, not necessarily surjective)
    vector<size_t> actualRowForExpRow(E, 0);
    {
        vector<bool> rowAssigned(E, false);
        decltype(E) rowsAssigned = 0;
        // now the pairs are sorted with closest string pairs first
        // just greedily assign the expected table rows to the actual rows, by closes string matching
        for (const pair<size_t, size_t>& matchings : indexPairs) {
            auto actualRowIdx = matchings.first;
            auto expectedRowIdx = matchings.second;
            if (!rowAssigned[expectedRowIdx]) {
                // assign it
                actualRowForExpRow[expectedRowIdx] = actualRowIdx;
                rowAssigned[expectedRowIdx] = true;
                rowsAssigned++;
                if (rowsAssigned == E) {
                    // finished assigning all rows in ground truth table
                    break;
                }
            }
        }
    }

    // now go through each matching and add up the levenshtein distance for the whole row
    if (expected.columns != actual.columns) {
        printf("Warning: expected table has %ld cols but actual table has %ld cols\n", expected.columns, actual.columns);
    }

    // levenshtein distance
    vector<double> keyColScores; // holds levenshtein distance for the between first columns (keys) in each row
    vector<double> avgValueColScores; // holds the average levenshtein distance for the remaining columns in each row
    {
        keyColScores.reserve(E);
        avgValueColScores.reserve(E);
        for (decltype(E) expectedRowIdx = 0; expectedRowIdx < E; ++expectedRowIdx) {
            const auto actualRow = actual.getRow(actualRowForExpRow[expectedRowIdx]);
            const auto expectedRow = expected.getRow(expectedRowIdx);
            keyColScores.push_back(levenshteinScore(actualRow[0], expectedRow[0]));

            // find the average levenshtein distance, assuming that the value columns match up
            // if the actual and expected tables have a different number of columns,
            // pretend that the table with fewer columns has just empty strings in those extra columns

            auto numValueColumns = std::max(actual.columns, expected.columns) - 1; // guaranteed no underflow due to earlier check
            double avgValueColScore;
            if (numValueColumns == 0) {
                avgValueColScore = 1.0; // nothing to compare, so perfect score
            } else {
                /* Compare as many columns as we can: min(actual.columns, expected.columns) - 1
                 * The remaining columns implicitly get a score of 0, because we use
                 *    numValueColumns = max(actual.columns, expected.columns) - 1 (which is > 0)
                 * as the normalising constant
                 */
                avgValueColScore = 0.0;
                for (decltype(numValueColumns) j = 1; j < std::min(actual.columns, expected.columns); ++j) {
                    avgValueColScore += levenshteinScore(actualRow[j], expectedRow[j]) / numValueColumns;
                }
            }
            avgValueColScores.push_back(avgValueColScore);
        }
    }


    const double sumKeyColScores = std::accumulate(keyColScores.begin(), keyColScores.end(), 0.0);
    // average levenshtein distance score in key column
    double avgKeyColScore = sumKeyColScores/E;
    // sum of levenshtein distance scores in value column;
    // weighted according to how well the corresponding key column matches the ground truth
    // the point of doing this is to try to minimise 'doubly penalising' badly matched key columns
    double weightedAvgValueColumnScore = 0.0;

    for (decltype(E) i = 0; i < E; ++i) {
        auto rowWeight = keyColScores[i]/sumKeyColScores;
        weightedAvgValueColumnScore += rowWeight*avgValueColScores[i];
    }

    return {avgKeyColScore, weightedAvgValueColumnScore};
}
