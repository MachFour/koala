//
// Created by max on 9/11/18.
//

#ifndef REFERENCE_TABLE_H
#define REFERENCE_TABLE_H

#include <iostream>
#include <vector>
#include <string>

class Table {
public:
    Table(size_t maxColumns) : maxColumns(maxColumns) {};

    void print(unsigned int minColumnWidth) const {
        // takes into account padding characters

        int realMinColWidth = minColumnWidth - 2;
        printf("\n\n");
        for (const std::vector<std::string>& column : rows) {
            for (const std::string& cell : column) {
                if (!cell.empty()) {
                    printf("%s", cell.data());
                }
                int fillChars = max(0, realMinColWidth - (unsigned int) cell.length());
                for (int i = 0; i < fillChars; ++i) {
                    putchar(' ');
                }
                putchar('|');
                putchar(' ');
            }
            printf("\n");
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
};

#endif //REFERENCE_TABLE_H
