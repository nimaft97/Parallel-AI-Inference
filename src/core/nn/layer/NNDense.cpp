#include "NNDense.h"

NNDense::NNDense(dimType units): m_units(units), m_weights({units,}), m_biases({units,})
{
}

NNTensor NNDense::forward(const NNTensor& input) const
{
    NNTensor output = NNTensor({m_units});
    return output;
}