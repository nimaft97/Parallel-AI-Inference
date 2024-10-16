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
    // Start with 0 index.
    dimType idx = 0;
    
    // Calculate the flattened index from multi-dimensional indices.
    for (dimType i = 0; i < indices.size(); ++i)
    {
        // Add the current index, scaled by the product of the subsequent dimensions.
        dimType scale = 1;
        for (dimType j = i + 1; j < m_dimensions.size(); ++j)
        {
            scale *= m_dimensions[j];
        }
        idx += indices[i] * scale;
    }

    return m_data[idx];
}


const float& Tensor::operator()(const dimVec& indices) const
{
    // Start with 0 index.
    dimType idx = 0;
    
    // Calculate the flattened index from multi-dimensional indices.
    for (dimType i = 0; i < indices.size(); ++i)
    {
        // Add the current index, scaled by the product of the subsequent dimensions.
        dimType scale = 1;
        for (dimType j = i + 1; j < m_dimensions.size(); ++j)
        {
            scale *= m_dimensions[j];
        }
        idx += indices[i] * scale;
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

Tensor Tensor::operator*(const Tensor& other) const
{
    const auto other_dims = other.getDimensions();
    if (other_dims.size() != 2 || m_dimensions.size() != 2)
    {
        throw std::invalid_argument("Only matrices can be multiplied");
    }

    if (other_dims[0] != m_dimensions[1])
    {
        throw std::invalid_argument("Matrix dimensions don't match the requested multiplication");
    }

    Tensor result = Tensor({m_dimensions[0], other_dims[1]});
    for (auto i = 0u; i < m_dimensions[0]; ++i)
    {
        for (auto j = 0u; j < other_dims[1]; ++j)
        {
            auto sum = 0.0f;
            for (auto k = 0u; k < m_dimensions[1]; ++k)
            {
                sum += (*this)({i, k}) * other({k, j});
            }
            result({i, j}) = sum;
        }
    }

    return result;
}