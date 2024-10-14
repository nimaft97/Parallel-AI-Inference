#include "NNTensor.h"

#include <numeric>

NNTensor::NNTensor(const dimVec& dimensions): m_dimensions(dimensions)
{
    m_size = std::accumulate(dimensions.cbegin(), dimensions.cend(), 1u, std::multiplies<dimType>());
    m_data.resize(m_size, 0.0f);
}

NNTensor::NNTensor(const NNTensor& other): 
                      m_dimensions(other.m_dimensions)
                    , m_size(other.m_size)
                    , m_data(other.m_data)
{
}

NNTensor::~NNTensor()
{
}

float& NNTensor::operator()(const dimVec& indices)
{
    dimType idx = 0;
    for (auto i = 0; i <= indices.size(); ++i)
    {
        idx *= m_dimensions[i];
        idx += indices[i];
    }

    return m_data[idx];
}

const float& NNTensor::operator()(const dimVec& indices) const
{
    dimType idx = 0;
    for (auto i = 0; i <= indices.size(); ++i)
    {
        idx *= m_dimensions[i];
        idx += indices[i];
    }

    return m_data[idx];
}

dimVec NNTensor::getDimensions() const
{
    return m_dimensions;
}

dimType NNTensor::getSize() const
{
    return m_size;
}