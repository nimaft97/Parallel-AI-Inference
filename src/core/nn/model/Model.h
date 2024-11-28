#ifndef MODEL_H
#define MODEL_H

#include "../layer/Layer.h"

#include <vector>
#include <string>
#include <memory>

class Model
{
public:
    Model();
    virtual void add_layer(Layer* p_layer);
    virtual void execute(const Tensor<float>* input, Tensor<float>* result1);
    virtual void to_host();
    virtual void to_device();
protected:
    std::vector<Layer*> m_layers;
    PLATFORM m_platform = PLATFORM::UNKNOWN;
};

#endif