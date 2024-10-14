#ifndef NNACTIVATION_H
#define NNACTIVATION_H

#include "../tensor/NNTensor.h"

class NNActivation
{
public:
    static NNTensor relu(const NNTensor& input);
    static NNTensor softmax(const NNTensor& input);
};

#endif