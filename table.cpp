//
// Created by max on 10/1/18.
//

#include "table.h"

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

void Table::checkCol(int col) const {
    if (col < 0 || col >= columns) {
        throw std::invalid_argument("column index out of range");
    }
}

void Table::checkRow(int row) const {
    if (row < 0 || row >= numRows()) {
        throw std::invalid_argument("row index out of range");
    }
}

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

int Table::numRows() const {
    return static_cast<int>(rows.size());
}

int Table::numCols() const {
    return columns;
}

// returns empty string if indices are out of range
string Table::getText(int row, int col) const {
    checkCol(col);
    checkRow(row);

    return rows[row][col];
}

const stringVector& Table::getRow(int row) const {
    checkRow(row);

    return rows[row];
}

void Table::addRow() {
    rows.emplace_back(stringVector((size_t)columns));
}
void Table::setColumnText(int row, int col, const std::string& text) {
    checkCol(col);

    auto rowSize = static_cast<size_t>(row);
    while (rowSize >= rows.size()) {
        addRow();
    }
    rows[row][col] = text;
}

Table Table::parseFromString(string tableString, string columnSep) {
    stringVector rowStrings = split(tableString, "\n");
    std::vector<stringVector> columnSplits;
    columnSplits.reserve(rowStrings.size());

    int maxColumns = 0;
    for (const string& rowString : rowStrings) {
        if (rowString.find(columnSep) == std::string::npos) {
            // no column separators so it's probably garbage; ignore it
            continue;
        }
        stringVector cells = split(rowString, columnSep);
        maxColumns = max(maxColumns, (int)cells.size());
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
