#ifndef COMMON_H
#define COMMON_H

#include <cassert>
#include <cstring>
#include <fstream>
#include <streambuf>

enum PLATFORM
{
    UNKNOWN = 0,
    HOST,
    DEVICE

};

std::string read_file(const std::string& file_path);

#endif  // COMMON_H