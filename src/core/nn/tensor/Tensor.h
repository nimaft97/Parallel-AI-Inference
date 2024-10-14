#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

typedef unsigned int dimType;
typedef std::vector<dimType> dimVec;

class Tensor
{
public:
    Tensor(const dimVec& dimensions);
    Tensor(const Tensor& other);
    ~Tensor();

    dimVec getDimensions() const;
    dimType getSize() const;
    float getMax() const;

    float& operator()(const dimVec& indices);
    const float& operator()(const dimVec& indices) const;
    float& operator()(const dimType& flattened_index);
    const float& operator()(const dimType& flattened_index) const;

private:
    std::vector<float> m_data;
    dimVec m_dimensions;
    dimType m_size;
};

#endif