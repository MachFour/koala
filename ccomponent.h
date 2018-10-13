//
// Created by max on 8/5/18.
//

#ifndef REFERENCE_CCOMPONENT_H
#define REFERENCE_CCOMPONENT_H

#include "wordBB.h"
/*
 * Class to store all the connected component details/stats, as returned by
 * opencv's connectedComponentsWithStats()
 */
class CComponent {
public:

    CComponent() : CComponent(0, 0, 0, 0) {}

    CComponent(int left, int top, int width, int height) :
            area(0), centroidX(0), centroidY(0), label(0),
            box(left, top, width, height) {}
    int area;
    double centroidX;
    double centroidY;
    int label;

    int left() const { return box.left(); }
    int right() const { return box.right(); }
    int top() const { return box.top(); }
    int bottom() const { return box.bottom(); }
    int width() const { return box.width(); }
    int height() const { return box.height(); }

    int boxArea() const { return box.boxArea(); }
    double boxAspectRatio() const { return box.boxAspectRatio(); }

    const wordBB& getBox() const {
        return box;
    }

private:
    wordBB box;
};

#endif //REFERENCE_CCOMPONENT_H
