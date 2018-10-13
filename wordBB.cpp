//
// Created by max on 10/13/18.
//


#include "wordBB.h"

void wordBB::expandHeightPx(int px) {
    _top -= px / 2;
    _height += px;

}

void wordBB::expandWidthPx(int px) {
    _left -= px / 2;
    _width += px;

}

void wordBB::expandMinOf(int absolute, double relative) {
    auto expandW = std::min(absolute, (int) std::round(relative * _width));
    auto expandH = std::min(absolute, (int) std::round(relative * _height));
    // XXX this may lead to sometimes each dimension getting expanded by the same amount,
    // sometimes different
    expandWidthPx(expandW);
    expandHeightPx(expandH);

}

void wordBB::constrain(int cLeft, int cTop, int cRight, int cBottom) {
    auto newLeft = std::max(left(), cLeft);
    auto newTop = std::max(top(), cTop);
    auto newRight = std::min(right(), cRight);
    auto newBottom = std::min(bottom(), cBottom);

    _left = newLeft;
    _top = newTop;
    _width = newRight - newLeft;
    _height = newBottom - newTop;
}


cv::Scalar wordBB::getColour(bool useRowColumn) const {
    if (useRowColumn) {
        return pseudoRandomColour(colAssigned ? _col : _row);
    } else {
        return pseudoRandomColour(_left, _top, _width, _height);
    }
}

int wordBB::left() const {
    return _left;
}

int wordBB::top() const {
    return _top;
}

int wordBB::width() const {
    return _width;
}

int wordBB::height() const {
    return _height;
}

int wordBB::row() const {
    return _row;
}

int wordBB::col() const {
    return _col;
}

int wordBB::right() const {
    return _left + _width;
}

int wordBB::bottom() const {
    return _top + _height;
}

void wordBB::setRow(int row) {
    _row = row;
    rowAssigned = true;
}

void wordBB::setCol(int col) {
    _col = col;
    colAssigned = true;
}

bool wordBB::isRowAssigned() const {
    return rowAssigned;
}

bool wordBB::isColAssigned() const {
    return colAssigned;
}

const std::string& wordBB::text() const {
    return _text;
}

void wordBB::setText(const std::string& text) {
    _text = text;
}

int wordBB::boxArea() const {
    return _width * _height;
}
double wordBB::boxCentreX() const {
    return _left + _width / 2.0;
}
double wordBB::boxCentreY() const {
    return _top + _height / 2.0;
}

//  height, return 0
// width : height. Returns zero if height is zero
double wordBB::boxAspectRatio() const {
    return (_height != 0) ? _width / (double) _height : 0.0;
}

cv::Rect wordBB::asRect() const {
    return cv::Rect(_left, _top, _width, _height);
}

using std::vector;
using std::min;
using std::max;

wordBB wordBB::combineAll(const vector<wordBB>& toCombine) {
    auto top = std::numeric_limits<int>::max(); // initially >= anything inside image
    auto left = std::numeric_limits<int>::max();  // initially >= anything inside image
    auto bottom = 0;         // initially <= anything inside image
    auto right = 0;         // initially <= anything inside image
    for (const auto& w : toCombine) {
        top = min(w.top(), top);
        left = min(w.left(), left);
        bottom = max(w.bottom(), bottom);
        right = max(w.right(), right);
    }
    auto height = bottom - top;
    auto width = right - left;

    return wordBB(left, top, width, height);
}

#include "IntervalGroup.tpp"

vector<wordBB> wordBB::combineHorizontallyClose(const vector<wordBB>& toCombine, double expand, Interval::ExpandType expandType) {
    std::function<Interval(wordBB)> intervalFunc = [](const wordBB& w) -> Interval { return Interval(w.left(), w.right()); };

    auto closeGroups = Interval::groupClose(toCombine, expand, expandType, intervalFunc);
    vector<wordBB> combined;
    for (const vector<wordBB>& group : closeGroups) {
        combined.push_back(wordBB::combineAll(group));
    }
    return combined;

}
