#include <NgraphNetworkCreator.hpp>
#include <OperationsFactory.hpp>
#define LOG_TAG "NgraphNetworkCreator"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

NgraphNetworkCreator::NgraphNetworkCreator(NnapiModelInfo* model, const std::string& plugin)
    : mModelInfo(model),
      mNgraphNodes(std::make_shared<NgraphNodes>(mModelInfo->getModel().operands.size())),
      mOpFctryInst(plugin, mNgraphNodes) {
    ALOGD("NgraphNetworkCreator Constructed");
}

std::map<OperationType, std::shared_ptr<OperationsBase>> OperationsFactory::mOperationsMap;

bool NgraphNetworkCreator::init() {
    ALOGI("%s", __func__);
    return true;
}

InferenceEngine::CNNNetwork* NgraphNetworkCreator::generateIRGraph() {
    ALOGI("%s", __func__);

    auto operations = mModelInfo->getOperations();
    for (const auto& op : operations) {
        auto nGraphOp = NgraphOpsFactory::createNgraphOp(op.type, mModelInfo, this);
        if (!nGraphOp->createNode(op)) {
            ALOGE("Failed to createNode for op type:%d", op.type);
            return nullptr;
        }
    }

    ngraph::OutputVector opVec;
    for (auto iter = mNgraphResultNodes.begin(); iter != mNgraphResultNodes.end(); iter++) {
        opVec.push_back(iter->second);
    }

    ngraph::ParameterVector inVec;
    for (auto iter = mNgraphInputNodes.begin(); iter != mNgraphInputNodes.end(); iter++) {
        inVec.push_back(iter->second);
    }

    auto net = new InferenceEngine::CNNNetwork(std::make_shared<ngraph::Function>(opVec, inVec));
    return net;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
