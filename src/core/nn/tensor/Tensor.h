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

    virtual dimVec getDimensions() const;
    virtual dimType getSize() const;
    virtual float getMax() const;
    virtual void setData(const std::vector<float>& data);

    virtual float& operator()(const dimVec& indices);
    virtual const float& operator()(const dimVec& indices) const;
    virtual float& operator()(const dimType& flattened_index);
    virtual const float& operator()(const dimType& flattened_index) const;
    virtual Tensor operator*(const Tensor& other) const;

protected:
    std::vector<float> m_data;
    dimVec m_dimensions;
    dimType m_size;
};

#endif