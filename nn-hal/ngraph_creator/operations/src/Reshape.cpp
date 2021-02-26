#include <NgraphNetworkCreator.hpp>
#include <Reshape.hpp>

#define LOG_TAG "ReshapeOperation"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
bool Reshape::validate(const Operation& op, NnapiModelInfo* modelInfo) {
    ALOGV("Entering %s", __func__);

    const auto& inputOperand = modelInfo->getOperand(op.inputs[0]);
    const auto& outputShapeOperand = modelInfo->getOperand(op.inputs[1]);
    const auto& outputOperand = modelInfo->getOperand(op.outputs[0]);

    if (outputOperand.type != OperandType::TENSOR_FLOAT32) {
        ALOGE("NNERR:output operand types invalid,aborting!!");
        return false;
    }

    if (outputOperand.type != OperandType::TENSOR_FLOAT32) {
        ALOGE("NNERR:input operand types invalid,aborting!!");
        return false;
    }

    if (outputShapeOperand.type != OperandType::INT32) {
        ALOGE("NNERR:output shape types invalid,aborting!!");
        return false;
    }
    // TODO:add check for output shape special value -1
    return true;
}

bool Reshape::createNode(const Operation& nnApiOp) {
    std::shared_ptr<ngraph::Node> inputNode = nullptr, shapeNode = nullptr;
    bool special_zero;
    auto createNode = [&](Operation op, uint32_t index) -> std::shared_ptr<ngraph::Node> {
        auto inputIndex = op.inputs[index];
        ngraph::Shape inShape;
        auto nnOperand = mModelInfo->getOperand(inputIndex);

        ALOGD("Input index: %d type: %d", inputIndex, nnOperand.type);
        if (nnOperand.lifetime == OperandLifeTime::MODEL_INPUT) {
            std::string name = "Reshape-" + std::to_string(mNwCreator->getNumber());
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
        } else if ((nnOperand.lifetime == OperandLifeTime::CONSTANT_COPY) ||
                   (nnOperand.lifetime == OperandLifeTime::CONSTANT_REFERENCE)) {
            ALOGD("Input is of type : const copy / reference %d", nnOperand.dimensions.size());
            auto vals = mModelInfo->GetConstVecOperand<float>(inputIndex);

            for (auto val : vals) {
                ALOGD("Dumping vals: %f", val);
            }
            auto in = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::f32, ngraph::Shape(toNgraphShape(nnOperand.dimensions)), vals);
            return in;
        } else {
            ALOGD("Input is of type temporary variable or unsupported");
            return nullptr;
        }
    };
    auto getNode = [&](uint32_t index) {
        std::shared_ptr<ngraph::Node> node;
        uint32_t outIndex;
        std::tie(node, outIndex) = mNwCreator->getIntermediateNodeOutput(index);
        return node->outputs()[outIndex];
    };
    ALOGD("========> Creating input node");
    inputNode = createNode(nnApiOp, 0);
    ALOGD("========> Creating shape node");
    shapeNode = createNode(nnApiOp, 1);

    auto shapeIndex = op.inputs[1];
    auto shapeOperand = mModelInfo->getOperand(shapeIndex);

    if (shapeIndex.dimensions[0] == 0) {
        special_zero = true;
    } else {
        special_zero = false;
    }

    std::shared_ptr<ngraph::Node> reshapeNode;
    reshapeNode = std::make_shared<ngraph::opset3::Reshape>(inputNode, shapeNode, special_zero);

    auto outputName = reshapeNode->outputs()[0].get_node()->get_friendly_name();
    ALOGD("Output name: %s", outputName.c_str());

    // Check if the output is output node or intermediate node in the graph
    switch (mModelInfo->getOperandLifetime(nnApiOp.outputs[0])) {
        case OperandLifeTime::TEMPORARY_VARIABLE:
            ALOGD("Output lifetime TEMPORARY_VARIABLE");
            mNwCreator->addIntermediateNode(nnApiOp.outputs[0], reshapeNode->outputs()[0]);
            mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[0], reshapeNode, 0);
            break;
        case OperandLifeTime::MODEL_OUTPUT:
            ALOGD("Output lifetime MODEL_OUTPUT");
            mNwCreator->addResultNode(nnApiOp.outputs[0], reshapeNode);
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