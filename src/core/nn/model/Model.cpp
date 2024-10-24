#include "Model.h"

#include <iostream>

Model::Model()
{
}

Model::~Model()
{
}

void Model::loadModel(const std::string& file_path)
{
    std::cout << "Loading model from: " << file_path << std::endl;
}

Tensor Model::runInference(const Tensor& input) const
{
    Tensor output = Tensor(input);

    for (const auto& layer : m_layers)
    {
        output = layer->forward(output);
    }
    return output;

}