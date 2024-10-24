#include "Dense.h"

Dense::Dense(dimType units): m_units(units), m_weights({units,}), m_biases({units,})
{
}

Tensor Dense::forward(const Tensor& input) const
{
    Tensor output = Tensor({m_units});
    return output;
}