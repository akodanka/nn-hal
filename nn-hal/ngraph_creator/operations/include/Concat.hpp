#ifndef __CONCAT_H
#define __CONCAT_H

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Concat : public OperationsBase {
public:
    Concat(NnapiModelInfo* model, NgraphNetworkCreator* nwCreator)
        : OperationsBase(model, nwCreator) {}
    static bool validate(const Operation& op, NnapiModelInfo* modelInfo);
    bool createNode(const Operation& operation) override;
    virtual ~Concat() {}
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif