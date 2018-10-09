//
// Created by max on 8/5/18.
//

#ifndef REFERENCE_CCOMPONENT_H
#define REFERENCE_CCOMPONENT_H

/*
 * Class to store all the connected component details/stats, as returned by
 * opencv's connectedComponentsWithStats()
 */
struct CComponent {
    int left;
    int top;
    int width;
    int height;
    int area;
    double aspectRatio;
    double centroidX;
    double centroidY;
    int label;
};

#endif //REFERENCE_CCOMPONENT_H
