#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <numeric>

enum PLATFORM
{
    UNKNOWN = 0,
    HOST,
    DEVICE

};

template<typename DATA_T>
class Tensor
{
public:
    Tensor(const Tensor<DATA_T>& other);
    Tensor(const std::vector<DATA_T>& h_data);

    virtual PLATFORM get_platform() const;
    virtual size_t get_size() const;
    virtual size_t get_dims() const;


    template<typename... Args>
    const DATA_T& operator(Args... indices) const
    {
        if (!is_indices_valid(std::forward(indices)))
        {
            throw std::invalid_argument("Passed indices are not valid");
        }
        const auto indices_vec = std::vector<size_t><{static_cast<size_t>(indices)...}>;
        const auto index = calculate_index(indices_vec);
        return  m_host_data[index];      
    }
    template<typename... Args>
    DATA_T& operator(Args... indices)
    {
        if (!is_indices_valid(std::forward(indices)))
        {
            throw std::invalid_argument("Passed indices are not valid");
        }
        const auto indices_vec = std::vector<size_t><{static_cast<size_t>(indices)...}>;
        const auto index = calculate_index(indices_vec);
        return  m_host_data[index];      
    }

    virtual void set_host_data(const std::vector<DATA_T>& h_data);
    virtual void set_dims(const std::vector<size_t>& dims);
    virtual void load_to_device() = 0;
    virtual void load_to_host() = 0;

    virtual Tensor<DATA_T> operator+(const Tensor<DATA_T>& other) const;
    virtual Tensor<DATA_T> operator*(const Tensor<DATA_T>& other) const;
protected:
    virtual Tensor<DATA_T> add_on_host(const Tensor<DATA_T>& other) const;
    virtual Tensor<DATA_T> add_on_device(const Tensor<DATA_T>& other) const = 0;
    virtual Tensor<DATA_T> multiply_on_host(const Tensor<DATA_T>& other) const;
    virtual Tensor<DATA_T> multiply_on_device(const Tensor<DATA_T>& other) const = 0;
private:
    size_t calculate_index(const std::vector<size_t>& indices) const;
    template<typename... Args>
    bool is_indices_valid(Args... indices) const
    {
        return (sizeof...(indices) == dims.size());
    }

public:
protected:
    std::vector<DATA_T> m_host_data;                    // vector on the host side
    std::vector<size_t> m_dims;                         // number of dimensions
    size_t              m_size;                         // number of elements
    PLATFORM            m_platform = PLATFORM::UNKNOWN;     // which device data is laoded to
private:
};

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
    m_size = std::accumulate(dims.cbegin(), dims.cend(), 1u, std::multiply<size_t>());
}

template<typename DATA_T>
void Tensor<DATA_T>::set_host_data(const std::vector<DATA_T>& h_data)
{
    m_host_data = h_data;
    m_size = h_data.size();
    if (m_dims.empty())
    {
        // assume it's a vector of dims are not set yet
        m_dims = {m_size, 1u};
    }
}

template<typename DATA_T>
Tensor<DATA_T> Tensor<DATA_T>::operator+(const Tensor<DATA_T>& other) const
{
    const auto this_device = get_platform();
    const auto other_device = other.get_platform();

    if (this_device != other_device)
    {
        throw std::invalid_argument("Both tensors must have their data on the same platform");
    }

    Tensor<DATA_T> result;

    switch(this_device)
    {
        case PLATFORM::HOST:
            result = add_on_host(other);
            break;
        case PLATFORM::DEVICE:
            result = add_on_device(other);
            break;
        default:
            std::cerr << "Data must either reside on host or device" << std::endl;
            break; 
    }
    return result;
}

template<typename DATA_T>
Tensor<DATA_T> Tensor<DATA_T>::operator*(const Tensor<DATA_T>& other) const
{
    const auto this_device = get_platform();
    const auto other_device = other.get_platform();

    if (this_device != other_device)
    {
        throw std::invalid_argument("Both tensors must have their data on the same platform");
    }

    if (m_dims != 2 || other.m_dims != 2 || m_dims[1] != other.m_dims[0])
    {
        throw std::invalid_argument("Dimensions are not feasible for matrix multiplication");
    }

    Tensor<DATA_T> result;

    switch(this_device)
    {
        case PLATFORM::HOST:
            result = multiply_on_host(other);
            break;
        case PLATFORM::DEVICE:
            result = multiply_on_device(other);
            break;
        default:
            std::cerr << "Data must either reside on host or device" << std::endl;
            break; 
    }
    return result;
}

#endif  // TENSOR_H