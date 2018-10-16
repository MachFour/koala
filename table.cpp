//
// Created by max on 10/1/18.
//

#include "table.h"

#include <vector>
#include <string>
#include <iterator>
#include <numeric>
#include <algorithm>

using std::string;
using std::vector;

static vector<string> split(const string& test, const string& delim);

// adapted from JDługosz's answer at
// https://codereview.stackexchange.com/questions/193203/splitting-stdstring-based-on-delimiter-using-only-find-and-substr

void Table::checkCol(size_t col) const {
    if (col >= cols()) {
        throw std::invalid_argument("column index out of range");
    }
}
void Table::checkRow(size_t row) const {
    if (row >= rows()) {
        throw std::invalid_argument("row index out of range");
    }
}

// column separators become '\f' characters
std::string Table::parseableString(const char * colSep) const {
    // takes into account padding characters
    std::string outStr;
    for (const vector<string>& column : _rows) {
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
    string outStr;
    outStr.reserve(minColumnWidth* cols()* rows());
    // takes into account padding characters
    const int realMinColWidth = minColumnWidth - 2;
    for (const auto& column : _rows) {
        for (const auto& cellText : column) {
            outStr.append(cellText);
            auto fillChars = static_cast<size_t>(std::max(0, realMinColWidth - (int) cellText.length()));
            outStr.append(fillChars, ' ');
            outStr.append("| ");
        }
        outStr.append("\n");
    }
    outStr.shrink_to_fit();
    return outStr;
}

size_t Table::rows() const {
    return _rows.size();
}

size_t Table::cols() const {
    return _columns;
}

string Table::getText(size_t row, size_t col) const {
    checkCol(col);
    checkRow(row);

    return _rows[row][col];
}

const vector<string>& Table::getRow(size_t row) const {
    checkRow(row);

    return _rows[row];
}

void Table::addRow() {
    _rows.emplace_back(vector<string>((size_t)_columns));
}
void Table::setText(size_t row, size_t col, const std::string &text) {
    checkCol(col);

    while (row >= _rows.size()) {
        addRow();
    }
    _rows[row][col] = text;
}

Table Table::parseFromString(string tableString, string columnSep) {
    vector<string> rowStrings = split(tableString, "\n");
    vector<vector<string>> rows;
    rows.reserve(rowStrings.size());

    size_t maxColumns = 0;
    for (const string& rowString : rowStrings) {
        if (rowString.find(columnSep) == std::string::npos) {
            // no column separators so it's probably garbage; ignore it
            continue;
        }
        vector<string> columns = split(rowString, columnSep);
        maxColumns = std::max(maxColumns, columns.size());
        rows.push_back(columns);
    }

    // ensure all rows have the same number of columns
    for (auto& row : rows) {
        row.resize(maxColumns);
    }
    rows.shrink_to_fit();

    // donate rows to table
    return Table(maxColumns, rows);
}

int Table::columnDifference(const Table &t1, const Table &t2) {
    return static_cast<int>(t1.cols()) - static_cast<int>(t2.cols());
}

// adapted from JDługosz's answer at
// https://codereview.stackexchange.com/questions/193203/splitting-stdstring-based-on-delimiter-using-only-find-and-substr

static vector<string> split(const string& test, const string& delim) {
    auto nextChar = test.cbegin();
    const auto stringEnd = test.cend();
    const auto delimStart = delim.cbegin();
    const auto delimEnd = delim.cend();
    const auto delimLength = delim.length();

    vector<string> splitStrings;

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
