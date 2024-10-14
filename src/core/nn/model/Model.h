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
    Model();
    ~Model();
    void loadModel(const std::string& file_path);
    Tensor runInference(const Tensor& input) const;


protected:
    std::vector<std::unique_ptr<Layer>> m_layers;
};

#endif