#pragma once

//#include <Driver.h>
#include <android/log.h>
#include <log/log.h>
#include <NgraphNodes.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>

#include <ModelManager.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class OperationsBase {
protected:
    std::shared_ptr<NnapiModelInfo> mModelInfo;
    std::shared_ptr<NgraphNodes> mNgraphNodes;
    enum ConversionType { NHWC_NCHW, NCHW_NHWC };
    std::shared_ptr<ngraph::Node> transpose(ConversionType type,
                                            ngraph::Output<ngraph::Node> input);
    virtual std::shared_ptr<ngraph::Node> createNode(const Operation& op) = 0;
    // override createNodeForPlugin in case sPluginType specific implementation is required
    virtual std::shared_ptr<ngraph::Node> createNodeForPlugin(const Operation& op);

public:
    static std::string sPluginType;
    OperationsBase(NnapiModelInfo* model);
    void setNgraphNodes(std::shared_ptr<NgraphNodes> nodes);
    virtual bool validate(const Operation& op);
    // override connectOperationToGraph in case Operation has multiple outputs
    virtual void connectOperationToGraph(const Operation& op);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
