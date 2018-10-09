//
// Created by max on 9/11/18.
//

// don't build this on android
#ifndef REFERENCE_ANDROID

#include "plotutils.h"

using cv::plot::Plot2d;

cv::Ptr<Plot2d> makePlot(const cv::Mat &data, const cv::Mat *resize, cv::Scalar colour, int thickness) {
    if (resize != nullptr) {
        return makePlot(data, colour, thickness, resize->cols, resize->rows);
    } else {
        return makePlot(data, colour, thickness);
    }
}

cv::Ptr<Plot2d> makePlot(const cv::Mat &data, cv::Scalar colour, int thickness, int resizeWidth, int resizeHeight) {
    cv::Ptr<Plot2d> plot = Plot2d::create(data);
    plot->setNeedPlotLine(true);
    plot->setShowGrid(false);
    plot->setPlotLineWidth(thickness);
    plot->setPlotLineColor(colour);
    if (resizeWidth >= 0 && resizeHeight >= 0) {
        plot->setPlotSize(resizeWidth, resizeHeight);
    }
    plot->setInvertOrientation(true);

    return plot;
}

#endif
