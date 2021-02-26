#ifndef __FULLY_CONNECTED_H
#define __FULLY_CONNECTED_H

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// create an FullyConnected Node based on the arguments/parameters.
class FullyConnected : public OperationsBase {
public:
    FullyConnected(NnapiModelInfo* model, NgraphNetworkCreator* nwCreator)
        : OperationsBase(model, nwCreator) {}
    static bool validate(const Operation& op, NnapiModelInfo* modelInfo);
    bool createNode(const Operation& operation) override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif