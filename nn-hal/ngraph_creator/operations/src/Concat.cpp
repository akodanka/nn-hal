#include <Concat.hpp>
#include <NgraphNetworkCreator.hpp>

#define LOG_TAG "ConcatOperation"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

bool Concat::validate(const Operation& op, NnapiModelInfo* modelInfo) {
    ALOGV("Entering %s", __func__);
    int op_size = op.inputs.size();
    ALOGD("Convolution input size = %d\n", op_size);

    // Check Output type
    const auto& outputOperand = modelInfo->getOperand(op.outputs[0]);

    if (outputOperand.type != OperandType::TENSOR_FLOAT32) {
        ALOGE("NNERR:output operand types invalid,aborting!!");
        return false;
    }

    // check concatenation axis
    const auto& concatAxis = modelInfo->getOperand(op.inputs[op_size - 1]);

    if (concatAxis.type != OperandType::INT32) {
        ALOGE("NNERR:invalid concatenation axis,aborting!!");
        return false;
    }
    // TODO: add check for 0-(n-1) input operandtype
    return true;
}

bool Concat::createNode(const Operation& nnApiOp) {
    auto n = nnApiOp.inputs.size() - 1;

    // //TODO:check with anoob
    // std::vector<uint32_t> axisMap = {2, 3, 1};  // NCHW = axisMap[NHWC]
    // auto axis = axisMap[mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, n)];

    auto axis = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, n);

    // std::vector<ngraph::Output<ngraph::Node>> inputs;
    std::vector<std::shared_ptr<ngraph::Node>> inputs;
    ALOGD("createNode n %d, axis %d", n, axis);

    auto createNode = [&](Operation op,
                          uint32_t index) -> std::shared_ptr<ngraph::Node> {
        auto inputIndex = op.inputs[index];
        ngraph::Shape inShape;
        auto nnOperand = mModelInfo->getOperand(inputIndex);

        ALOGD("Input index: %d type: %d", inputIndex, nnOperand.type);
        if (nnOperand.lifetime == OperandLifeTime::MODEL_INPUT) {
            std::string name = "Concat-" + std::to_string(mNwCreator->getNumber());
            ALOGD("Input is of type model input %s  type=%d", name.c_str(), nnOperand.type);
            auto in = std::make_shared<ngraph::opset3::Parameter>(
                ngraph::element::f32, toNgraphShape(nnOperand.dimensions));
            in->set_friendly_name(name);

            ALOGD("Setting input layer name: %s", name.c_str());
            mNwCreator->addInputNode(inputIndex, in);

            ALOGD("Adding layer metadata");
            mNwCreator->addLayerMetadata(inputIndex, LayerInfo(name, false), true);

            ALOGD("Done ...........");
            return in;
        } else {
            ALOGD("Input is of type temporary variable or unsupported");
            return nullptr;
        }
    };

    auto getNode = [&](uint32_t index) {
        std::shared_ptr<ngraph::Node> node;
        uint32_t outIndex;
        std::tie(node, outIndex) =
            mNwCreator->getIntermediateNodeOutput(index);
        return node->outputs()[outIndex];
    };

    for (int i = 0; i < n; i++) {
        std::shared_ptr<ngraph::Node> inputNode = nullptr;
        inputNode = createNode(nnApiOp, i);
        inputs.push_back(inputNode);
    }

    std::shared_ptr<ngraph::Node> concatNode;
    concatNode = std::make_shared<ngraph::opset3::Concat>(inputs, axis);

    auto outputName = concatNode->outputs()[0].get_node()->get_friendly_name();
    ALOGD("Output name: %s", outputName.c_str());

    // Check if the output is output node or intermediate node in the graph
    switch (mModelInfo->getOperandLifetime(nnApiOp.outputs[0])) {
        case OperandLifeTime::TEMPORARY_VARIABLE:
            ALOGD("Output lifetime TEMPORARY_VARIABLE");
            mNwCreator->addIntermediateNode(nnApiOp.outputs[0], concatNode->outputs()[0]);
            mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[0], concatNode, 0);
            break;
        case OperandLifeTime::MODEL_OUTPUT:
            ALOGD("Output lifetime MODEL_OUTPUT");
            mNwCreator->addResultNode(nnApiOp.outputs[0], concatNode);
            mNwCreator->addLayerMetadata(nnApiOp.outputs[0], LayerInfo(outputName, false), false);
            break;
        default:
            ALOGE("Unsupported lifetime for output node: %d",
                  mModelInfo->getOperandLifetime(nnApiOp.outputs[0]));
            break;
    }
    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
