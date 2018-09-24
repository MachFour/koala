//
// Created by max on 9/11/18.
//

#ifndef REFERENCE_TABLE_H
#define REFERENCE_TABLE_H

#include <vector>
#include <string>

#include <cstdio>
#ifdef REFERENCE_ANDROID
#include <android/log.h>
#endif

class Table {
public:
    Table(size_t maxColumns) : maxColumns(maxColumns) {};

    // column separators become '\f' characters
    std::string parseableString(const char * colSep = "\f") const {
        // takes into account padding characters
        std::string outStr;
        for (const std::vector<std::string>& column : rows) {
            for (const std::string& cell : column) {
                outStr.append(cell);
                outStr.append(colSep);
            }
            // replace last column separator with newline to mark end of row
            outStr.back() = '\n';
        }
        return outStr;
    }

    std::string printableString(unsigned int minColumnWidth) const {
        std::string outStr("\n\n");
        // takes into account padding characters
        int realMinColWidth = minColumnWidth - 2;
        for (const std::vector<std::string>& column : rows) {
            for (const std::string& cell : column) {
                outStr.append(cell);
                auto fillChars = rectifiedDifference(realMinColWidth, (int) cell.length());
                outStr.append(fillChars, ' ');
                outStr.append("| ");
            }
            outStr.append("\n");
        }
        return outStr;
    }

    size_t numRows() const {
        return rows.size();
    }

    // returns empty string if indices are out of range
    std::string getText(unsigned int row, unsigned int col) {
        if (row >= numRows() || col >= maxColumns) {
            return std::string("");
        } else {
            return rows[row][col];
        }
    }

    void addRow() {
        std::vector<std::string> row(maxColumns);
        rows.push_back(row);
    }
    void setColumnText(unsigned int row, unsigned int col, std::string text) {
        while (row >= rows.size()) {
            addRow();
        }
        rows[row][col] = text;
    }

private:
    size_t maxColumns;
    std::vector<std::vector<std::string>> rows;

    static unsigned int rectifiedDifference(int a, int b) {
        return a - b <= 0 ? 0 : (unsigned int)(a - b);
    }
};

#endif //REFERENCE_TABLE_H
