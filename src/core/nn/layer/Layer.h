#ifndef LAYER_H
#define LAYER_H

#include "../tensor/Tensor.h"

class Layer
{
public:
    Layer();
    virtual Tensor<float> forward(const Tensor<float>& input) const = 0;
    virtual PLATFORM get_platform() const;
    // virtual void to_device() = 0;
    // virtual void to_host() = 0;
protected:
    PLATFORM m_platform = PLATFORM::UNKNOWN;
};

#endif