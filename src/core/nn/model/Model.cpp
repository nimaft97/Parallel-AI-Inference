#include "Model.h"

Model::Model(): m_layers()
{
}

void Model::add_layer(const Layer* p_layer)
{
    m_layers.emplace_back(p_layer);
}

Tensor<float> Model::execute(const Tensor<float>& input)
{
    Tensor<float> result = input;

    for (auto& p_layer : m_layers)
    {
        result = p_layer->forward(result);
    }
    return result;
}