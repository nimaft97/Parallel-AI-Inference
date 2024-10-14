#ifndef NNMODEL_H
#define NNMODEL_H

#include "../tensor/NNTensor.h"
#include "../layer/NNLayer.h"

#include <vector>
#include <string>
#include <memory>

class NNModel
{
public:
    NNModel();
    ~NNModel();
    void loadModel(const std::string& file_path);
    NNTensor runInference(const NNTensor& input) const;


protected:
    std::vector<std::unique_ptr<NNLayer>> m_layers;
};

#endif