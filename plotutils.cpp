//
// Created by max on 9/11/18.
//

// don't build this on android
#ifndef REFERENCE_ANDROID

#include "plotutils.h"

cv::Ptr<Plot> makePlot(const cv::Mat &data, const cv::Mat *resize, cv::Scalar colour, int thickness) {
    cv::Ptr<Plot> plot = Plot::create(data);
    plot->setNeedPlotLine(true);
    plot->setShowGrid(false);
    plot->setPlotLineWidth(thickness);
    plot->setPlotLineColor(colour);
    if (resize != nullptr) {
        plot->setPlotSize(resize->cols, resize->rows);
    }
    plot->setInvertOrientation(true);

    return plot;
}

#endif
