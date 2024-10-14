#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../tensor/Tensor.h"

class Activation
{
public:
    static Tensor relu(const Tensor& input);
    static Tensor softmax(const Tensor& input);
};

#endif