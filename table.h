//
// Created by max on 9/11/18.
//

#ifndef REFERENCE_TABLE_H
#define REFERENCE_TABLE_H

#include <vector>
#include <string>
#include <iterator>
#include <algorithm>

#ifdef REFERENCE_ANDROID
#include <android/log.h>
#endif


class Table {
    using string = std::string;
    using stringVector = std::vector<std::string>;

public:
    Table(size_t columns) : columns(columns) {};

    std::string parseableString(const char * colSep = "\f") const;

    string printableString(unsigned int minColumnWidth) const;
    size_t numRows() const;

    // returns empty string if indices are out of range
    string getText(size_t row, size_t col) const;
    const stringVector& getRow(size_t row) const;


    void addRow();
    void setColumnText(size_t row, size_t col, std::string text);

    static Table parseFromString(string tableString, string columnSep="\f");
    static auto compareTable(const Table& actual, const Table& expected) -> std::pair<double, double>;

private:
    size_t columns;
    std::vector<std::vector<std::string>> rows;
};

#endif //REFERENCE_TABLE_H
