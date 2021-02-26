#include <NgraphNetworkCreator.hpp>
#include <Relu.hpp>

#define LOG_TAG "ReluOperation"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
bool Relu::validate(const Operation& op, NnapiModelInfo* modelInfo) { return true; }

bool Relu::createNode(const Operation& nnApiOp) {
    // addNode(nodeName, std::make_shared<ngraph::opset3::Relu>(mNodes[inputName]));
    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android