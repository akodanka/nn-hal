#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Add : public OperationsBase {
public:
    Add(std::shared_ptr<NnapiModelInfo> model);
    bool validate(const Operation& op) override;
    std::shared_ptr<ngraph::Node> createNode(const Operation& operation) override;
    std::shared_ptr<ngraph::Node> createNodeForPlugin(const Operation& operation) override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
