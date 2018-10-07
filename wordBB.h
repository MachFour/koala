//
// Created by max on 8/18/18.
//

#ifndef REFERENCE_WORDBB_H
#define REFERENCE_WORDBB_H

#include "randomColour.h"
#include <string>
#include <opencv2/core.hpp>

/*
 * Holds data related to expanded bounding boxes, used to track candidate words
 * Has fields to facilitate clustering into rows and columns
 */

struct wordBB {
    int x;
    int y;
    int width;
    int height;
    std::string text;

    explicit wordBB(const cv::Rect& r) : wordBB(r.x, r.y, r.width, r.height) {};

    wordBB(int x, int y, int width, int height) :
        x(x), y(y), width(width), height(height), text(""),
        _row(0), _col(0), rowAssigned(false), colAssigned(false) {};


    int row() const {
        return _row;
    }
    int col() const {
        return _col;
    }

    void setRow(int row) {
        _row = row;
        rowAssigned = true;
    }
    void setCol(int col) {
        _col = col;
        colAssigned = true;
    }
    bool isRowAssigned() const {
        return rowAssigned;
    }
    bool isColAssigned() const {
        return colAssigned;
    }

    int getArea() const {
        return width*height;
    }

    cv::Rect asRect() const {
        return cv::Rect(x, y, width, height);
    }

    cv::Scalar getColour(bool useColumn=false) const {
        if (useColumn) {
            return pseudoRandomColour(_col);
        } else {
            return pseudoRandomColour(x, y, width, height);
        }
    }
private:
    int _row;
    int _col;
    bool rowAssigned;
    bool colAssigned;

};

#endif //REFERENCE_WORDBB_H
