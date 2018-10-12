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

class wordBB {
public:
    int left;
    int top;
    int width;
    int height;
    std::string text;

    explicit wordBB(const cv::Rect& r) : wordBB(r.x, r.y, r.width, r.height) {};

    wordBB(int left, int top, int width, int height) :
        left(left), top(top), width(width), height(height), text(""),
        _row(0), _col(0), rowAssigned(false), colAssigned(false) {};

    cv::Scalar getColour(bool useRowColumn=false) const;

    /*
     * Mutator methods
     */

    // WARNING these use integer division so expandHeight(x); expandHeight(-x) may shift the thing by one pixel down, idk
    void expandHeightPx(int px);
    void expandWidthPx(int px);
    void expandMinOf(int pixels, int percent);
    void constrain(int cLeft, int cTop, int cRight, int cBottom);

    int row() const {
        return _row;
    }
    int col() const {
        return _col;
    }

    int right() const {
        return left + width;
    }

    int bottom() const {
        return top + height;
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
        return cv::Rect(left, top, width, height);
    }

private:
    int _row;
    int _col;
    bool rowAssigned;
    bool colAssigned;

};

#endif //REFERENCE_WORDBB_H
