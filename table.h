//
// Created by max on 9/11/18.
//

#ifndef REFERENCE_TABLE_H
#define REFERENCE_TABLE_H

#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <stdexcept>

class Table {
    using string = std::string;
    using stringVector = std::vector<std::string>;

public:
    explicit Table(int columns) : columns(columns) {
        if (columns < 0) {
            throw std::invalid_argument("Number of columns must be positive");
        }
    }

    std::string parseableString(const char * colSep = "\f") const;

    string printableString(unsigned int minColumnWidth) const;
    int numRows() const;
    int numCols() const;

    // returns empty string if indices are out of range
    string getText(int row, int col) const;
    const stringVector& getRow(int row) const;


    void addRow();
    void setColumnText(int row, int col, const std::string& text);

    static Table parseFromString(string tableString, string columnSep="\f");

private:
    // bounds checking
    void checkCol(int col) const;
    void checkRow(int row) const;

    int columns;
    std::vector<stringVector> rows;
};

#endif //REFERENCE_TABLE_H
