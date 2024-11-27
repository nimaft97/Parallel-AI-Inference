#ifndef TENSOR_H
#define TENSOR_H

#include "../common.h"

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <memory>

template<typename DATA_T>
class Tensor
{
public:
    Tensor(const Tensor<DATA_T>& other);
    Tensor();
    virtual ~Tensor() = default;

    virtual PLATFORM get_platform() const;
    virtual size_t get_size() const;
    virtual const std::vector<size_t>& get_dims() const;

    template<typename... Args>
    const DATA_T& operator()(Args... indices) const
    {
        if (!is_indices_valid(std::forward<Args>(indices)...))
        {
            std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
            throw std::invalid_argument("Passed indices are not valid");
        }
        const auto indices_vec = std::vector<size_t>({static_cast<size_t>(indices)...});
        const auto index = calculate_index(indices_vec);
        return  m_host_data[index];      
    }
    template<typename... Args>
    DATA_T& operator()(Args... indices)
    {
        if (!is_indices_valid(std::forward<Args>(indices)...))
        {
            std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
            throw std::invalid_argument("Passed indices are not valid");
        }
        const auto indices_vec = std::vector<size_t>({static_cast<size_t>(indices)...});
        const auto index = calculate_index(indices_vec);
        return  m_host_data[index];      
    }

    virtual void set_host_data(const std::vector<DATA_T>& h_data);
    virtual void set_dims(const std::vector<size_t>& dims);
    virtual void load_to_device();
    virtual void load_to_host();

    virtual void add(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const;
    virtual void multiply(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const;

    virtual std::string to_string(bool platform=true, bool dim=true, bool total_size=true, bool data=false) const;
    virtual Tensor<DATA_T>* clone() const;

protected:
    virtual void add_on_host(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const;
    virtual void multiply_on_host(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const;
    virtual void add_on_device(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const;
    virtual void multiply_on_device(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const;

    virtual bool is_operation_valid(const Tensor<DATA_T>* left, const Tensor<DATA_T>* right, const Tensor<DATA_T>* result, PLATFORM platform) const;
private:
    size_t calculate_index(const std::vector<size_t>& indices) const;
    template<typename... Args>
    bool is_indices_valid(Args... indices) const
    {
        return (sizeof...(indices) == m_dims.size());
    }

public:
protected:
    std::vector<DATA_T> m_host_data;                        // vector on the host side
    std::vector<size_t> m_dims;                             // number of dimensions
    size_t              m_size;                             // number of elements
    PLATFORM            m_platform = PLATFORM::UNKNOWN;     // which device data is loaded to
private:
};

template<typename DATA_T>
Tensor<DATA_T>::Tensor(): m_host_data({}), m_dims({}), m_size(0), m_platform(PLATFORM::UNKNOWN)
{
}

template<typename DATA_T>
Tensor<DATA_T>::Tensor(const Tensor<DATA_T>& other)
{
    m_host_data = other.m_host_data;
    m_dims      = other.m_dims;
    m_size      = other.m_size;
    m_platform  = other.m_platform;
}

template<typename DATA_T>
Tensor<DATA_T>* Tensor<DATA_T>::clone() const
{
    return new Tensor<DATA_T>(*this);
}

template<typename DATA_T>
PLATFORM Tensor<DATA_T>::get_platform() const
{
    return m_platform;
}

template<typename DATA_T>
size_t Tensor<DATA_T>::get_size() const
{
    return m_size;
}

template<typename DATA_T>
const std::vector<size_t>& Tensor<DATA_T>::get_dims() const
{
    return m_dims;
}

template<typename DATA_T>
void Tensor<DATA_T>::set_dims(const std::vector<size_t>& dims)
{
    m_dims = dims;
    m_size = std::accumulate(dims.cbegin(), dims.cend(), 1u, std::multiplies<size_t>());
}

template<typename DATA_T>
void Tensor<DATA_T>::set_host_data(const std::vector<DATA_T>& flattened_data)
{
    m_platform = PLATFORM::HOST;
    m_host_data = flattened_data;
    m_size = flattened_data.size();

    size_t num_elements_from_dim = m_dims.empty() ? 0 : std::accumulate(m_dims.cbegin(), m_dims.cend(), 1u, std::multiplies<size_t>());
    // if m_dims is empty or old
    if (num_elements_from_dim != m_size)
    {
        // assume data is 1D
        m_dims = {m_size, 1u};
    }
}

template<typename DATA_T>
bool Tensor<DATA_T>::is_operation_valid(const Tensor<DATA_T>* left, const Tensor<DATA_T>* right, const Tensor<DATA_T>* result, PLATFORM platform) const
{
    const auto this_plat   = left->get_platform();
    const auto other_plat  = right->get_platform();
    const auto result_plat = result->get_platform();

    if (!(this_plat == other_plat && this_plat == result_plat && this_plat == platform))
    {
        return false;
    }
    return true;
}

template<typename DATA_T>
void Tensor<DATA_T>::add(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const
{
    if (!is_operation_valid(this, other, result, m_platform))
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::invalid_argument("Not all tensors are on the same platform");
    }
    
    // check if dimensions are valid
    const auto other_dims = other->get_dims(); 
    if (!(m_dims.size() == 2 && other_dims.size() == 2 && m_dims[0] == other_dims[0] && m_dims[1] == other_dims[1]))
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::runtime_error("Invalid dimensions");
    }

    result->set_dims(m_dims);

    switch (m_platform)
    {
        case PLATFORM::HOST:
            add_on_host(other, result);
            break;
        case PLATFORM::DEVICE:
            add_on_device(other, result);
            break;
        default:
            std::cerr << "Unsupported platform!";
    }
}

template<typename DATA_T>
void Tensor<DATA_T>::multiply(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const
{
    if (!is_operation_valid(this, other, result, m_platform))
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::invalid_argument("Not all tensors are on the same platform");
    }
    
    // check if dimensions are valid
    const auto other_dims = other->get_dims(); 
    if (!(m_dims.size() == 2 && other_dims.size() == 2 && m_dims[1] == other_dims[0]))
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::runtime_error("Invalid dimensions");
    }

    result->set_dims({m_dims[0], other_dims[1]});

    switch (m_platform)
    {
        case PLATFORM::HOST:
            multiply_on_host(other, result);
            break;
        case PLATFORM::DEVICE:
            multiply_on_device(other, result);
            break;
        default:
            std::cerr << "Unsupported platform!";
    }
}

template<typename DATA_T>
void Tensor<DATA_T>::add_on_host(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const
{
    // std::transform(m_host_data.begin(), m_host_data.begin() + get_size(), other->m_host_data.begin(), result->m_host_data.begin(), std::plus<DATA_T>());
    for (auto i = 0u; i < m_dims[0]; ++i)
    {
        for (auto j = 0u; j < m_dims[1]; ++j)
        {
            (*result)(i, j) = (*this)(i, j) + (*other)(i, j);
        }
    }
}

template<typename DATA_T>
void Tensor<DATA_T>::multiply_on_host(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const
{
    for (auto i = 0u; i < m_dims[0]; ++i)
    {
        for (auto j = 0u; j < other->m_dims[1]; ++j)
        {
            DATA_T sum = static_cast<DATA_T>(0);
            for (auto k = 0u; k < m_dims[1]; ++k)
            {
                sum += (*this)(i, k) * (*other)(k, j);
            }
            (*result)(i, j) = sum;
        }
    }
}

template<typename DATA_T>
void Tensor<DATA_T>::add_on_device(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const
{
    // to be overwritten by derived classes if needed
}

template<typename DATA_T>
void Tensor<DATA_T>::multiply_on_device(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const
{
    // to be overwritten by derived classes if needed
}

template<typename DATA_T>
std::string Tensor<DATA_T>::to_string(bool platform, bool dim, bool total_size, bool data) const
{
    std::ostringstream oss;

    if (platform)
    {
        oss << "platform: ";
        switch (m_platform)
        {
            case PLATFORM::HOST:
                oss << "Host";
                break;
            case PLATFORM::DEVICE:
                oss << "Device";
                break;
            default:
                oss << "Unknown";
                break;
        }
        oss << " ";
    }

    if (dim)
    {
        oss << "dim: {";
        for (auto i = 0; i < m_dims.size(); ++i)
        {
            oss << m_dims[i];
            if (i != m_dims.size() - 1)
            {
                oss << ", ";
            }
        }
        oss << "} ";
    }

    if (total_size)
    {
        oss << "size: " << m_size << " ";
    }

    if (data)
    {
        oss << "host data: ";
        if (m_platform != PLATFORM::HOST)
        {
            oss << "not on host. ";
        }
        else if (m_dims.size() > 2)
        {
            oss << "only matrix data is supported.";
        }
        else
        {
            oss << "{";
            for (auto i = 0u; i < m_dims[0]; ++i)
            {
                oss << "{";
                for (auto j = 0u; j < m_dims[1]; ++j)
                {
                    oss << (*this)(i, j);
                    if (j != m_dims[1] - 1)
                    {
                        oss << ", ";
                    }
                }
                oss << "}";
                if (i != m_dims[0] - 1)
                {
                    oss << ", ";
                }
            }
            oss << "}";
        }
    }

    if (platform || dim || total_size || data)
    {
        oss << "\n";
    }

    return oss.str();
}

template<typename DATA_T>
size_t Tensor<DATA_T>::calculate_index(const std::vector<size_t>& indices) const
{
    // Start with 0 index.
    size_t idx = 0;
    
    // Calculate the flattened index from multi-dimensional indices.
    for (auto i = 0u; i < indices.size(); ++i)
    {
        // Add the current index, scaled by the product of the subsequent dimensions.
        size_t scale = 1;
        for (auto j = i + 1; j < m_dims.size(); ++j)
        {
            scale *= m_dims[j];
        }
        idx += indices[i] * scale;
    }
    return idx;
}

template<typename DATA_T>
void Tensor<DATA_T>::load_to_host()
{
    m_platform = PLATFORM::HOST;
}

template<typename DATA_T>
void Tensor<DATA_T>::load_to_device()
{
    m_platform = PLATFORM::DEVICE;
}

#endif  // TENSOR_H