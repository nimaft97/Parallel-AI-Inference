#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"

class Dense : public Layer
{
public:

    virtual void forward(const Tensor<float>* input, Tensor<float>* result1, Tensor<float>* result2) const override;
    virtual void to_device() override;
    virtual void to_host() override;
    virtual void set_weight(Tensor<float>* weight);
    virtual void set_bias(Tensor<float>* bias);

protected:
    Tensor<float>* m_weight;
    Tensor<float>* m_bias;
};

#endif