#ifndef LAYER_H
#define LAYER_H

#include "../tensor/Tensor.h"
#include "../common.h"

class Layer
{
public:
    virtual Tensor<float> forward(const Tensor<float>& input) const = 0;
    virtual PLATFORM get_platform() const;
    virtual void load_to_device() = 0;
    virtual void load_to_host() = 0;
protected:
    PLATFORM m_platform = PLATFORM::UNKNOWN;
};

#endif