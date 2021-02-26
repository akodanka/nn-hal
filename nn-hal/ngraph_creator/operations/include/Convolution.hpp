#ifndef __CONVOLUTION_H
#define __CONVOLUTION_H

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// To create an Convolution Node based on the arguments/parameters.
class Convolution : public OperationsBase {
public:
    Convolution(NnapiModelInfo* model, NgraphNetworkCreator* nwCreator)
        : OperationsBase(model, nwCreator) {}

    static bool validate(const Operation& op, NnapiModelInfo* modelInfo);
    bool createNode(const Operation& operation) override;
    virtual ~Convolution() {}
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif