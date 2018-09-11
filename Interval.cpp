//
// Created by max on 8/2/18.
//

#include <vector>
#include <algorithm>
#include "Interval.h"

template <typename T>
using vector = std::vector<T>;

bool Interval::areClose(const Interval &i1, const Interval &i2, double expand, ExpandType method) {
    double dist = distance(i1.midpoint, i2.midpoint);
    double longerHL = fmax(i1.halflength, i2.halflength);
    double shorterHL = fmin(i1.halflength, i2.halflength);
    switch (method) {
        case MIN:
            return dist < longerHL + shorterHL*expand*2;
        case MAX:
            return dist < longerHL*expand*2 + shorterHL;
        case AVG:
        default:
            return dist < longerHL*expand + shorterHL*expand;
    }
}

bool Interval::closeToAny(const Interval &one, const vector<Interval> &others, double expand, ExpandType method) {
    bool close = false;
    for (const Interval& other : others) {
        close |= areClose(one, other, expand, method);
        if (close) {
            // don't need to check others
            break;
        }
    }
    return close;
}

void Interval::groupCloseIntervals(vector<Interval> intervals, vector<vector<Interval>> &partitions, double expansion) {
    partitions.clear();

    // sort intervals by left edge, this way we can find overlapping intervals in one pass
    std::sort(intervals.begin(), intervals.end(), [](Interval a, Interval b) -> bool {
        return a.left() < b.left();
    });

    // since we sorted by left edge, adjacent intervals will all be contiguous in the intervals vector
    for (unsigned int i = 0; i < intervals.size(); ++i)  {
        // there are more intervals left, so add a new partition to the output array
        partitions.push_back(std::vector<Interval>());
        vector<Interval>& currentPartition = partitions.back();
        currentPartition.push_back(intervals[i]);
        // find how many subsequent intervals overlap
        // make sure not to skip over the i+1ths interval if it doesn't overlap
        for (; i+1 < intervals.size() && closeToAny(intervals[i + 1], currentPartition, expansion, MIN); ++i) {
            // TODO could interval i+1 be close to a previous interval in the current partition but not interval i?
            currentPartition.push_back(intervals[i+1]);
        }
    }
}
