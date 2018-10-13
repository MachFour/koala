//
// Created by max on 10/13/18.
//

#include "Interval.h"

#include <vector>
#include <functional>
#include <algorithm>

using std::function;

template <typename T>
vector<vector<T>> Interval::groupClose(const vector<T>& things, double expansion, ExpandType method, function<Interval(T)> makeInterval) {
    vector<Interval> intervals;
    intervals.reserve(things.size());
    for (size_t i = 0; i < things.size(); ++i) {
        Interval interval = makeInterval(things[i]);
        interval.setLabel((int)i);
        intervals.push_back(interval);
    }

    // sort by left edge of intervals, this way we can find overlapping intervals in one pass
    std::sort(intervals.begin(), intervals.end(), [](const Interval& i1, const Interval& i2) -> bool {
        return i1.left() < i2.left();
    });

    vector<vector<T>> partitions;
    // since we sorted by left edge, adjacent intervals will all be contiguous in the intervals vector
    for (size_t i = 0; i < intervals.size(); ++i)  {
        // there are more intervals left, so add a new partition to the output array
        vector<Interval> currentIntervals;
        vector<T> currentPartition;

        currentIntervals.push_back(intervals[i]);
        currentPartition.push_back(things[intervals[i].label()]);
        // find how many subsequent intervals overlap
        // make sure not to skip over the i+1ths interval if it doesn't overlap
        for (; i+1 < intervals.size(); ++i) {
            if (closeToAny(intervals[i + 1], currentIntervals, expansion, method)) {
                currentIntervals.push_back(intervals[i + 1]);
                currentPartition.push_back(things[intervals[i + 1].label()]);
            } else {
                break;
            }
        }
        partitions.push_back(currentPartition);
    }
    return partitions;
}
