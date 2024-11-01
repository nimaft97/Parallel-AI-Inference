#ifndef MODEL_H
#define MODEL_H

#include "../tensor/Tensor.h"
#include "../layer/Layer.h"

#include <vector>
#include <string>
#include <memory>

class Model
{
public:
protected:
    std::vector<std::unique_ptr<Layer>> m_layers;
};

#endif