#include "NNLayer.h"


// ****************************************** NNLayer ****************************************** 
NNLayer::NNLayer()
{
}

// ****************************************** NNDense ****************************************** 

NNDense::NNDense(dimType units): m_units(units), m_weights({units,}), m_biases({units,})
{
}

NNTensor NNDense::forward(const NNTensor& input) const
{
    NNTensor output = NNTensor({m_units});
    return output;
}

// ****************************************** NNConv2D ****************************************** 

NNConv2D::NNConv2D(dimType filters, dimType kernel_size, dimType stride, dimType padding)
    : m_filters(filters), m_kernel_size(kernel_size), m_stride(stride), m_padding(padding),
      m_weights({filters, kernel_size, kernel_size}), m_biases({filters})
{
}

NNTensor NNConv2D::forward(const NNTensor& input) const
{
    NNTensor output = NNTensor(input);
    return output;
}