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
    virtual void add_layer(const Layer* p_layer);
    virtual Tensor<float> execute(const Tensor<float>& input);
protected:
    std::vector<const Layer*> m_layers;
};

#endif