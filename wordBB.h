//
// Created by max on 8/18/18.
//

#ifndef REFERENCE_WORDBB_H
#define REFERENCE_WORDBB_H

#include "randomColour.h"
#include "Interval.h"
#include <string>
#include <vector>
#include <opencv2/core.hpp>

/*
 * Holds data related to expanded bounding boxes, used to track candidate words
 * Has fields to facilitate clustering into rows and columns
 */

class wordBB {
public:

private:
    int _left;
    int _top;
    int _width;
    int _height;
    int _row;
    int _col;
    bool rowAssigned;
    bool colAssigned;
    std::string _text;

public:

    explicit wordBB(const cv::Rect& r) : wordBB(r.x, r.y, r.width, r.height) {};

    wordBB(int left, int top, int width, int height) :
        _left(left), _top(top), _width(width), _height(height),
        _row(0), _col(0), rowAssigned(false), colAssigned(false), _text("") {};


    /*
     * Accessor methods
     */
    int left() const;
    int top() const;
    int right() const;
    int bottom() const;
    int width() const;
    int height() const;

    int row() const;
    int col() const;

    bool isRowAssigned() const;
    bool isColAssigned() const;

    const std::string& text() const;


    /*
     * Mutator methods
     */
    void setRow(int row);
    void setCol(int col);
    void setText(const std::string& text);


    // WARNING these use integer division so expandHeight(x); expandHeight(-x) may shift the thing by one pixel down, idk
    void expandHeightPx(int px);
    void expandWidthPx(int px);
    void expandMinOf(int absolute, double relative);
    void constrain(int cLeft, int cTop, int cRight, int cBottom);



    /*
     * Misc
     */
    cv::Scalar getColour(bool useRowColumn=false) const;
    cv::Rect asRect() const;
    int boxArea() const;
    double boxCentreX() const;
    double boxCentreY() const;
    double boxAspectRatio() const;

    static wordBB combineAll(const std::vector<wordBB>&);
    static std::vector<wordBB> combineHorizontallyClose(const std::vector<wordBB>&, double, Interval::ExpandType);
};

#endif //REFERENCE_WORDBB_H
