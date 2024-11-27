#include "Model.h"

Model::Model(): m_layers()
{
}

void Model::add_layer(const Layer* p_layer)
{
    m_layers.emplace_back(p_layer);
}

bool Model::execute(const Tensor<float>* input, Tensor<float>* result1) const
{
    const auto num_layers = m_layers.size();
    if (num_layers == 0)
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::runtime_error("Model does not have any layers");
    }

    auto result2 = result1->clone();
    auto result3 = result1->clone();
    
    m_layers[0]->forward(input, result1, result2);

    for (auto i = 1u; i < m_layers.size(); ++i)
    {
        // always the second argument would contain the result
        if (i % 3 == 1)
        {
            m_layers[i]->forward(result1, result2, result3);
        }
        else if (i % 3 == 2)
        {
            m_layers[i]->forward(result2, result3, result1);
        }
        else
        {
            m_layers[i]->forward(result3, result1, result2);
        }
    }

    if (m_layers.size() % 3 == 2)
    {
        *result1 = *result2->clone();
    }
    else if (m_layers.size() % 3 == 0)
    {
        *result1 = *result3->clone();
    }

    delete result2;
    delete result3;
}