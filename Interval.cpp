//
// Created by max on 8/2/18.
//

#include <vector>
#include <algorithm>
#include "Interval.h"


bool Interval::areClose(const Interval &i1, const Interval &i2, double expand, OverlapType method) {
    double dist = distance(i1.midpoint, i2.midpoint);
    switch (method) {
        case MIN:
            return dist < 2*fmin(i1.halflength, i2.halflength)*expand;
        case MAX:
            return dist < 2*fmax(i1.halflength, i2.halflength)*expand;
        case AVG:
        default:
            return dist < (i1.halflength + i2.halflength)*expand;
    }
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
        for (; i+1 < intervals.size() && areClose(intervals[i], intervals[i + 1], expansion, MIN); ++i) {
            currentPartition.push_back(intervals[i+1]);
        }
    }
}
