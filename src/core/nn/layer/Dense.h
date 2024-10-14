#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"

class Dense : public Layer
{
public:
    Dense(dimType units);
    Tensor forward(const Tensor& input) const override;

private:
    dimType m_units;
    Tensor m_weights;
    Tensor m_biases;
};

#endif