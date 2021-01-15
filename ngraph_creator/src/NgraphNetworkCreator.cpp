#include <NgraphNetworkCreator.hpp>
#define LOG_TAG "NgraphNetworkCreator"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
    

void NgraphNetworkCreator::createInputParams()
{
    for (auto i : mModel.inputIndexes) {
        std::shared_ptr<ngraph::opset3::Parameter> inputParam;
        auto& origDims = mModel.operands[i].dimensions;
        std::vector<size_t> dims(origDims.begin(), origDims.end());
        if(dims.size() == 3)
        {
            ALOGD("createInputParams converting operand %d to 4D", i);
            dims.insert(dims.begin(), 1);
        }
        switch(mModel.operands[i].type)
        {
            case OperandType::FLOAT32 :
            case OperandType::TENSOR_FLOAT32  :
                inputParam = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape(dims.begin(), dims.end()));
                ALOGD("createInputParams created inputIndex %d, type %d", i, mModel.operands[i].type);
                break;
            default :
                ALOGE("createInputParams Failure at inputIndex %d, type %d", i, mModel.operands[i].type);
                inputParam = nullptr;
        }
        mInputParams.push_back(inputParam);
        mOperationOutputs[i] = inputParam;
    }
}

NgraphNetworkCreator::NgraphNetworkCreator(const Model& model, const std::string& plugin) : mModel(model), mOpFctryInst(plugin)
{
    mOperationOutputs.reserve(mModel.operands.size());
    ALOGD("NgraphNetworkCreator Constructed");
}
    
bool NgraphNetworkCreator::validateOperations() {
    for (const auto& operation : mModel.operations) {
        if(!mOpFctryInst.getOperation(operation.type, mModel)->validate(operation))
            return false;
    }
    return true;
}

bool NgraphNetworkCreator::initializeModel() {
    int index = 0;
    createInputParams();
    for (const auto& operation : mModel.operations) {
        auto op = mOpFctryInst.getOperation(operation.type, mModel);
        if(op == nullptr)
        {
            ALOGD("initializeModel Failure at type %d", operation.type);
            return false;
        }
        mOperationOutputs[operation.outputs[0]] = //TODO: Handle multiple Outputs(eg.LSTM). Assumption here : each node has only 1 output.
            op->createNodeForPlugin(operation, mOperationOutputs);
    }
    ALOGD("initializeModel Success");
    return true;
}

std::string NgraphNetworkCreator::getNodeName(uint32_t index) {
    return mOperationOutputs[index]->get_name();
}

std::shared_ptr<ngraph::Function> NgraphNetworkCreator::generateGraph() {
    for (auto i : mModel.outputIndexes) {
        mResultNodes.push_back(mOperationOutputs[i]);
    }
    //mResultNodes.push_back(mOperationOutputs[mModel.outputIndexes[0]]);//Only output node to prevent disconnected graph
    mResultNodes.push_back(std::make_shared<ngraph::opset3::Concat>(mResultNodes, 1));//Dummy Concat to join the network
    return std::make_shared<ngraph::Function>(mResultNodes, mInputParams);
}

}
}
}
}
