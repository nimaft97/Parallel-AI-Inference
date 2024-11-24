#include "common.h"

std::string read_file(const std::string& file_path)
{
    std::ifstream file(file_path);
    assert(file.is_open() && "Couldn't open the file");
    // Read the entire file into a string
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}