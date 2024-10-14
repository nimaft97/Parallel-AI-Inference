#include "Tensor.h"

#include <numeric>
#include <algorithm>

Tensor::Tensor(const dimVec& dimensions): m_dimensions(dimensions)
{
    m_size = std::accumulate(dimensions.cbegin(), dimensions.cend(), 1u, std::multiplies<dimType>());
    m_data.resize(m_size, 0.0f);
}

Tensor::Tensor(const Tensor& other): 
                      m_dimensions(other.m_dimensions)
                    , m_size(other.m_size)
                    , m_data(other.m_data)
{
}

Tensor::~Tensor()
{
}

float& Tensor::operator()(const dimVec& indices)
{
    dimType idx = 0;
    for (auto i = 0; i <= indices.size(); ++i)
    {
        idx *= m_dimensions[i];
        idx += indices[i];
    }

    return m_data[idx];
}

const float& Tensor::operator()(const dimVec& indices) const
{
    dimType idx = 0;
    for (auto i = 0; i <= indices.size(); ++i)
    {
        idx *= m_dimensions[i];
        idx += indices[i];
    }

    return m_data[idx];
}

float& Tensor::operator()(const dimType& flattened_index)
{
    return m_data[flattened_index];
}

const float& Tensor::operator()(const dimType& flattened_index) const
{
    return m_data[flattened_index];
}

dimVec Tensor::getDimensions() const
{
    return m_dimensions;
}

dimType Tensor::getSize() const
{
    return m_size;
}

float Tensor::getMax() const
{
    return *std::max_element(m_data.data(), m_data.data()+m_size);
}