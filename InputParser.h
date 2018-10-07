//
// Created by max on 10/6/18.
//

#ifndef REFERENCE_INPUTPARSER_H
#define REFERENCE_INPUTPARSER_H

#include <string>
#include <algorithm>
#include <vector>

class InputParser {
public:
    InputParser(int argc, char ** argv);
    const std::string getCmdOption(const std::string &option) const;
    const std::string getArg(const std::size_t index) const;
    bool cmdOptionExists(const std::string &option) const;
private:
    std::vector<std::string> tokens;
};

#endif //REFERENCE_INPUTPARSER_H
