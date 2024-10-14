#ifndef NNTENSOR_H
#define NNTENSOR_H

#include <vector>

typedef unsigned int dimType;
typedef std::vector<dimType> dimVec;

class NNTensor
{
public:
    NNTensor(const dimVec& dimensions);
    NNTensor(const NNTensor& other);
    ~NNTensor();

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