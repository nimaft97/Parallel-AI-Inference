#include "NNActivation.h"

#include <cmath>

NNTensor NNActivation::relu(const NNTensor& input)
{
    NNTensor output = NNTensor(input.getDimensions());
    const auto total_size = input.getSize();
    for (auto flattened_idx = 0u; flattened_idx < total_size; ++flattened_idx)
    {
        const auto in_val = input(flattened_idx);  // read the ith element from m_data of NNTensor
        output(flattened_idx) = std::max(0.0f, in_val); 
    }
    return output;
}

NNTensor NNActivation::softmax(const NNTensor& input)
{
    NNTensor output = NNTensor(input.getDimensions());
    const auto max_in_val = input.getMax();
    const auto total_size = input.getSize();

    
    float sum = 0.0f;
    for (auto flattened_idx = 0u; flattened_idx < total_size; ++flattened_idx)
    {
        const auto val = std::exp(input(flattened_idx) - max_in_val);
        output(flattened_idx) = val;
        sum += val;
    }

    // normalize
    for (auto flattened_idx = 0u; flattened_idx < total_size; ++flattened_idx)
    {
        output(flattened_idx) /= sum;
    }

    return output;
}