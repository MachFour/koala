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
class CComponent : public wordBB {
public:

    CComponent() : CComponent(0, 0, 0, 0) {}

    CComponent(int left, int top, int width, int height) :
            wordBB(left, top, width, height),
            area(0), aspectRatio(0.0), centroidX(0), centroidY(0), label(0) {}
    int area;
    double aspectRatio;
    double centroidX;
    double centroidY;
    int label;
};

#endif //REFERENCE_CCOMPONENT_H
