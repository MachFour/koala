//
// Created by max on 10/22/18.
//

#ifndef KOALA_FILEUTILS_H
#define KOALA_FILEUTILS_H

#include <string>

std::string readFile(const std::string &filename);
std::string basename(std::string filename, bool removeExtension=false);

#endif //KOALA_FILEUTILS_H
