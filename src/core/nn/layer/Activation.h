#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Layer.h"

class Activation : public Layer
{
public:

    Activation(ACTIVATION activation);
    virtual void forward(const Tensor<float>* input, Tensor<float>* result1, Tensor<float>* result2) const override;
    virtual void to_device() override;
    virtual void to_host() override;
    virtual ACTIVATION get_activation() const;

protected:
    virtual void set_activation(ACTIVATION activation);

protected:
    ACTIVATION m_activation = ACTIVATION::UNKNOWN;
};

#endif