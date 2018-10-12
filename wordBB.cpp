//
// Created by max on 10/13/18.
//


#include "wordBB.h"

void wordBB::expandHeightPx(int px) {
    top -= px / 2;
    height += px;

}
void wordBB::expandWidthPx(int px) {
    left -= px / 2;
    width += px;

}

void wordBB::expandMinOf(int pixels, int percent) {
    auto expandW = std::min(pixels, (int)std::round(percent/100.0 * width));
    auto expandH = std::min(pixels, (int)std::round(percent/100.0 * height));
    // XXX this may lead to sometimes each dimension getting expanded by the same amount,
    // sometimes different
    expandWidthPx(expandW);
    expandHeightPx(expandH);

}

void wordBB::constrain(int cLeft, int cTop, int cRight, int cBottom) {
    auto newLeft = std::max(left, cLeft);
    auto newTop = std::max(top, cTop);
    auto newRight = std::min(left + width, cRight);
    auto newBottom = std::min(top + height, cBottom);

    this->left = newLeft;
    this->top = newTop;
    width = newRight - newLeft;
    height = newBottom - newTop;
}


cv::Scalar wordBB::getColour(bool useRowColumn) const {
    if (useRowColumn) {
        return pseudoRandomColour(colAssigned ? _col : _row);
    } else {
        return pseudoRandomColour(left, top, width, height);
    }
}
