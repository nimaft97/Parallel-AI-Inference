#ifndef NNDENSE_H
#define NNDENSE_H

#include "NNLayer.h"

class NNDense : public NNLayer
{
public:
    NNDense(dimType units);
    NNTensor forward(const NNTensor& input) const override;

private:
    dimType m_units;
    NNTensor m_weights;
    NNTensor m_biases;
};

#endif