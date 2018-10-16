//
// Created by max on 9/11/18.
//

#ifndef REFERENCE_TABLE_H
#define REFERENCE_TABLE_H

#include <vector>
#include <string>

class Table {

private:
    // steal rows object from caller
    Table(size_t columns, std::vector<std::vector<std::string>> rows) : _columns(columns), _rows(std::move(rows)) { }

public:
    explicit Table(size_t columns) : _columns(columns) { }
    explicit Table() : _columns(0) { }

    size_t rows() const;
    size_t cols() const;

    std::string getText(size_t row, size_t col) const;
    std::string parseableString(const char * colSep = "\f") const;
    std::string printableString(unsigned int minColumnWidth) const;

    const std::vector<std::string>& getRow(size_t row) const;

    void setText(size_t row, size_t col, const std::string &text);
    void addRow();

    static Table parseFromString(std::string tableString, std::string columnSep="\f");
    /*
     * Helper function to do appropriate casting
     * returns t1.numCols() - t2.cols()
     */
    static int columnDifference(const Table& t1, const Table& t2);

private:
    // bounds checking
    void checkCol(size_t col) const;
    void checkRow(size_t row) const;

    size_t _columns;
    std::vector<std::vector<std::string>> _rows;
};

#endif //REFERENCE_TABLE_H
