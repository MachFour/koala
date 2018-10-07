//
// Created by max on 10/6/18.
//

#include "InputParser.h"

/*
 * Implementation taken from
 * https://stackoverflow.com/questions/865668/how-to-parse-command-line-arguments-in-c
 * @author: iain
 */

InputParser::InputParser(int argc, char ** argv) {
    if (argc > 0) {
        tokens.reserve((std::size_t)argc);
        for (int i = 1; i < argc; ++i) {
            tokens.emplace_back(std::string(argv[i]));
        }
    }
}

const std::string InputParser::getArg(const std::size_t index) const {
    return (index < tokens.size()) ? tokens[index] : "";
}

const std::string InputParser::getCmdOption(const std::string &option) const {
    std::vector<std::string>::const_iterator itr;
    itr = find(tokens.begin(), tokens.end(), option);
    // if the option string was found, return the next string in the array
    // (i.e. the option argument)
    if (itr != tokens.end() && ++itr != tokens.end()) {
        return *itr;
    } else {
        return "";
    }
}

bool InputParser::cmdOptionExists(const std::string &option) const {
    return find(tokens.begin(), tokens.end(), option) != tokens.end();
}
