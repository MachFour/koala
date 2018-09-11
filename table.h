//
// Created by max on 9/11/18.
//

#ifndef REFERENCE_TABLE_H
#define REFERENCE_TABLE_H

#include <vector>
#include <string>

#ifndef REFERENCE_ANDROID
#include <cstdio>
#else
#include <android/log.h>
#endif

class Table {
public:
    Table(size_t maxColumns) : maxColumns(maxColumns) {};

    void print(unsigned int minColumnWidth) const {
        // takes into account padding characters

        int realMinColWidth = minColumnWidth - 2;
        outString("\n\n");
        for (const std::vector<std::string>& column : rows) {
            for (const std::string& cell : column) {
                if (!cell.empty()) {
                    outString(cell);
                }
                int fillChars = max(0, realMinColWidth - (unsigned int) cell.length());
                for (int i = 0; i < fillChars; ++i) {
                    outString(" ");
                }
                outString("| ");
            }
            outString("\n");
        }
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

    static int max(int a, int b) {
        return a >= b ? a : b;
    }

    static void outString(const std::string& s) {
#ifdef REFERENCE_ANDROID
        __android_log_write(ANDROID_LOG_DEBUG, "KTable", s.data());
#else
        printf("%s", s.data());
#endif
    }

};

#endif //REFERENCE_TABLE_H
