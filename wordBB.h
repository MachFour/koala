//
// Created by max on 8/18/18.
//

#ifndef REFERENCE_WORDBB_H
#define REFERENCE_WORDBB_H

#include <string>
#include <opencv2/core.hpp>
#include "randomColour.h"

/*
 * Holds data related to expanded bounding boxes, used to track candidate words
 * Has fields to facilitate clustering into rows and columns
 */

struct wordBB {

    wordBB(const cv::Rect& r) :
            wordBB(r.x, r.y, r.width, r.height) {};

    wordBB(int x, int y, int width, int height) :
        x(x), y(y), width(width), height(height), row(-1), column(-1), text("") {};

    int x;
    int y;
    int width;
    int height;

    int row;
    int column;
    std::string text;

    int getArea() const {
        return width*height;
    }

    cv::Rect asRect() const {
        return cv::Rect(x, y, width, height);
    }

    cv::Scalar getColour(bool useColumn=false) const {
        if (useColumn) {
            return pseudoRandomColour(column);
        } else {
            return pseudoRandomColour(x, y, width, height);
        }
    }
};

#endif //REFERENCE_WORDBB_H
