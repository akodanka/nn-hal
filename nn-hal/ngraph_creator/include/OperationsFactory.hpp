#ifndef __OPERATIONS_FACTORY_H
#define __OPERATIONS_FACTORY_H
#define LOG_TAG "OperationsFactory"
#include <Add.hpp>
#include <Concat.hpp>
#include <Convolution.hpp>
#include <DepthwiseConv.hpp>
#include <FullyConnected.hpp>
#include <Relu.hpp>
#include <Reshape.hpp>

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class OperationsFactory {
private:
    static std::map<OperationType, std::shared_ptr<OperationsBase>> mOperationsMap;
    // std::shared_ptr<NgraphNodes> mNgraphNodes;

public:
    // OperationsFactory(const std::string& plugin, std::shared_ptr<NgraphNodes> nodes);
    // std::shared_ptr<OperationsBase> getOperation(const OperationType& type, const Model& model);

    static std::unique_ptr<OperationsBase> createNgraphOp(OperationType opType,
                                                          NnapiModelInfo* modelInfo,
                                                          NgraphNetworkCreator* nwCreator) {
        ALOGD("%s", __func__);
        std::unique_ptr<OperationsBase> nodePtr = nullptr;
        switch (opType) {
            case OperationType::ADD:
                nodePtr = std::make_unique<Add>(modelInfo, nwCreator);
                // sOperationsMap[opType] = nodePtr;
                break;
            case OperationType::CONCATENATION:
                nodePtr = std::make_unique<Concat>(modelInfo, nwCreator);
                break;
            case OperationType::CONV_2D:
                nodePtr = std::make_unique<Convolution>(modelInfo, nwCreator);
                break;
            case OperationType::DEPTHWISE_CONV_2D:
                nodePtr = std::make_unique<DepthwiseConv>(modelInfo, nwCreator);
                break;
            case OperationType::FULLY_CONNECTED:
                nodePtr = std::make_unique<FullyConnected>(modelInfo, nwCreator);
                break;
            case OperationType::RELU:
                nodePtr = std::make_unique<Relu>(modelInfo, nwCreator);
                break;
            case OperationType::RESHAPE:
                nodePtr = std::make_unique<Reshape>(modelInfo, nwCreator);
                break;
            default:
                ALOGE("Operation of type: %d not supported", opType);
                break;
        }
        return nodePtr;
    }

    static bool isOperationSupported(Operation op, NnapiModelInfo* modelInfo) {
        ALOGD("%s", __func__);
        if (mOperationsMap.find(op.type) == mOperationsMap.end()) {
            switch (op.type) {
                case OperationType::ADD:
                    if (!Add::validate(op, modelInfo)) return false;
                    break;
                case OperationType::CONCATENATION:
                    if (!Concat::validate(op, modelInfo)) return false;
                    break;
                case OperationType::CONV_2D:
                    if (!Convolution::validate(op, modelInfo)) return false;
                    break;
                case OperationType::DEPTHWISE_CONV_2D:
                    if (!DepthwiseConv::validate(op, modelInfo)) return false;
                    break;
                case OperationType::FULLY_CONNECTED:
                    if (!FullyConnected::validate(op, modelInfo)) return false;
                    break;
                case OperationType::RELU:
                    if (!Relu::validate(op, modelInfo)) return false;
                    break;
                case OperationType::RESHAPE:
                    if (!Reshape::validate(op, modelInfo)) return false;
                    break;
                default:
                    ALOGE("Failed to validate operation: %d", op.type);
                    return false;
            }
        }

        // ALOGD("%s succeeded", __func__);
        return true;
    }
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif