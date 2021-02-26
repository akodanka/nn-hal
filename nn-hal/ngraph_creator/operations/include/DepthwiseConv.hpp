#ifndef __DWCONVOLUTION_H
#define __DWCONVOLUTION_H

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// To create an DepthwiseConv Node based on the arguments/parameters.
class DepthwiseConv : public OperationsBase {
public:
    DepthwiseConv(NnapiModelInfo* model, NgraphNetworkCreator* nwCreator)
        : OperationsBase(model, nwCreator) {}

    static bool validate(const Operation& op, NnapiModelInfo* modelInfo);
    bool createNode(const Operation& operation) override;
    virtual ~DepthwiseConv() {}
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif