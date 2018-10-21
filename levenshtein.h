
#ifndef LEVENSHTEIN_H
#define LEVENSHTEIN_H

#include <numeric>
#include <string>
#include <vector>

/*
 * Optimised levenshtein string distance function, by Frederik Hertzum.
 * Taken from https://bitbucket.org/clearer/iosifovich, on 24/09/18
 * Modified to allow compilation under C++11 and to simplify it a bit
 * Original function:

#pragma once

#include <iterator>
#include <numeric>
#include <vector>

auto levenshtein(std::string a, std::string b) -> size_t {
	auto [a_begin, b_end] = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
	auto i = std::distance(a.begin(), a_begin);

	size_t j = 0;
	while (j + i < std::min(a.length(), b.length()) && a[a.length() - j] == b[b.length() - j]) ++j;

	a = a.substr(i, a.length() - i);
	b = b.substr(i, b.length() - i);
	if (a.size() > b.size()) std::swap(a, b);

	std::vector<size_t> buffer(b.length() + 1);

	std::iota(buffer.begin(), buffer.end(), 0);
	for (size_t i = 1; i < a.length() + 1; ++i) {
		auto temp = buffer[0]++;
		for (size_t j = 1; j < buffer.size(); ++j) {
			auto p = buffer[j - 1];
			auto r = buffer[j];
			temp = std::min(std::min(r, p) + 1, temp + (a[i - 1] == b[j - 1] ? 0 : 1));
			std::swap(buffer[j], temp);
		}
	}
	return buffer.back();
}
 */

template <typename T>
auto min3(T a, T b, T c) -> T {
    using std::min;
    return min(min(a, b), c);
}

auto levenshtein(std::string a, std::string b) -> size_t {
    // don't worry about finding matching initial or final substrings
    // since they won't occur very often anyway, and the strings aren't too long

	// make b the longer string
	if (a.size() > b.size()) {
	    std::swap(a, b);
	}
	
	std::vector<size_t> buffer(b.length() + 1);
	// incremental assignment
	std::iota(buffer.begin(), buffer.end(), 0);
	for (size_t i = 1; i < a.length() + 1; ++i) {
		auto temp = buffer[0]++;
		for (size_t j = 1; j < buffer.size(); ++j) {
			auto p = buffer[j - 1];
			auto r = buffer[j];
			size_t editCost = a[i - 1] == b[j - 1] ? 0 : 1;
			temp = min3(r+1, p+1, temp + editCost);
			std::swap(buffer[j], temp);
		}
	}
	return buffer.back();
}

/*
 * Levenshtein distance 'Score'
 * Defined as follows: If |a| is the length of string a, and LD(a, b)
 * is the levenshtein distance function, then the score is:
 * S := 1 - LD(a, b)/max(|a|, |b|)
 *
 * Since LD(a, b) <= max(|a|, |b|), this is a rough way to create a
 * 'correctness' fraction using the Levenshtein distance function
 */
double stringSimilarity(const std::string &a, const std::string &b) {
	if (a.empty() && b.empty()) {
		return 1.0;
	} else if (a.empty() || b.empty()) {
		return 0.0;
	} else {
		return 1.0 - (double) levenshtein(a, b)/std::max(a.size(), b.size());
	}
}

/*
 * Asymmetric string similarity. String b is assumed to be ground truth string,
 * so its length is used to normalise the levenshtein distance.
 * Minimum score is still zero
 */
double asymStringSimilarity(const std::string &a, const std::string &b) {
	if (a.empty() && b.empty()) {
		return 1.0;
	} else if (a.empty() || b.empty()) {
		return 0.0;
	} else {
		return std::max(0.0, 1.0 - (double) levenshtein(a, b)/b.size());
	}
}

/*
 * Taken from Definition 3 of
 * "A Normalized Levenshtein Distance Metric"
 * Li Yujian and Liu Bo
 * IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
 */
double levenshteinSimilarity(const std::string &a, const std::string &b) {
	if (a.empty() && b.empty()) {
		return 1.0;
	} else if (a.empty() || b.empty()) {
		return 0.0;
	} else {
	    auto ld = levenshtein(a, b);
	    auto sumLengths = a.size() + b.size();
	    //return 1.0 - 2.0*ld / (a.size() + b.size() + ld);
		return (double) (sumLengths - ld) / (sumLengths + ld);
	}

}

#endif // LEVENSHTEIN_H
