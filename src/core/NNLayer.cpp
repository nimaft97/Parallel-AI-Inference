#include "NNLayer.h"


// ****************************************** NNLayer ****************************************** 
NNLayer::NNLayer()
{
}

NNLayer::~NNLayer()
{
}

// ****************************************** NNDense ****************************************** 

NNTensor NNDense::forward(const NNTensor& input) const
{
    return input;
}

// ****************************************** NNConv2D ****************************************** 

NNTensor NNConv2D::forward(const NNTensor& input) const
{
    return input;
}