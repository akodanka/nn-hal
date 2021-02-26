#ifndef __ADD_H
#define __ADD_H

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// create an Add Node based on the arguments/parameters.
class Add : public OperationsBase {
public:
    Add(NnapiModelInfo* model, NgraphNetworkCreator* nwCreator)
        : OperationsBase(model, nwCreator) {}
    static bool validate(const Operation& op, NnapiModelInfo* modelInfo);
    bool createNode(const Operation& operation) override;
    virtual ~Add() {}
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif