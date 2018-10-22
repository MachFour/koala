//
// Created by max on 10/22/18.
//

#include "fileutils.h"

#include <fstream>
//#include <iostream>
#include <ios>

// https://codereview.stackexchange.com/questions/22901/reading-all-bytes-from-a-file
// https://en.cppreference.com/w/cpp/io/basic_istream/read
std::string readFile(const std::string &filename) {
    // initially seek to end of file to get its position
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        fprintf(stderr, "Could not open file with name %s\n", filename.c_str());
        return "";
    }
    // XXX this may not always be an accurate indicator of the file size!
    std::fstream::pos_type end = ifs.tellg();
    if (end == std::fstream::pos_type(-1)) {
        // error
        fprintf(stderr, "readFile(): error end-of-file position: %s\n", filename.c_str());
        return "";
    }

    auto size = static_cast<size_t>(end);
#ifdef REFERENCE_ANDROID
#warning "tellg() may not work as a size indication on Android"
#endif

    std::string fileString(size, '\0');
    ifs.seekg(0);
    ifs.read(&fileString[0], size);
    ifs.close();

    return fileString;
}

std::string basename(std::string filename, bool removeExtension) {
    // check for and remove trailing slash(es)
    while (filename.back() == '/') {
        filename.pop_back();
    }
    size_t lastSlash = filename.rfind('/');
    if (lastSlash != std::string::npos) {
        // guaranteed that slash was not the last character, so there's at least one more character after lastSlash
        filename.erase(0, lastSlash+1);
    }
    size_t lastDot = filename.rfind('.');
    if (removeExtension && lastDot != std::string::npos) {
        filename.erase(lastDot, filename.size() - lastDot);
    }
    filename.shrink_to_fit();
    return filename;
}


