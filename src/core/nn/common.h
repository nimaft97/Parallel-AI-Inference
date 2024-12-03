#ifndef COMMON_H
#define COMMON_H

#include <cassert>
#include <cstring>
#include <fstream>
#include <streambuf>

enum class PLATFORM
{
    UNKNOWN = 0,
    HOST,
    DEVICE

};

enum class ACTIVATION
{
    UNKNOWN = 0,
    RELU
};

std::string read_file(const std::string& file_path);

#endif  // COMMON_H