#include "NNModel.h"

#include <iostream>

NNModel::NNModel()
{
}

NNModel::~NNModel()
{
}

void NNModel::loadModel(const std::string& file_path)
{
    std::cout << "Loading model from: " << file_path << std::endl;
}

NNTensor NNModel::runInference(const NNTensor& input) const
{
    NNTensor output = NNTensor(input);

    for (const auto& layer : m_layers)
    {
        output = layer->forward(output);
    }
    return output;

}