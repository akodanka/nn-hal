#ifndef __OPERATIONS_BASE_H
#define __OPERATIONS_BASE_H

#include <Driver.h>
#include <Temp.h>  //TODO: Remove this once NNAPI_Utils is ready
#include <android/log.h>
#include <log/log.h>
// #include <NgraphNodes.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/shape.hpp>

#include "ModelManager.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphNetworkCreator;

class OperationsBase {
protected:
    // Model mModel;
    // std::shared_ptr<NgraphNodes> mNgraphNodes;
    // enum ConversionType { NHWC_NCHW, NCHW_NHWC };
    // std::shared_ptr<ngraph::Node> transpose(ConversionType type,
    //                                         ngraph::Output<ngraph::Node> input);
    // virtual std::shared_ptr<ngraph::Node> createNode(const Operation& op) = 0;
    // // override createNodeForPlugin in case sPluginType specific implementation is required
    // virtual std::shared_ptr<ngraph::Node> createNodeForPlugin(const Operation& op);
    NnapiModelInfo* mModelInfo;
    NgraphNetworkCreator* mNwCreator;
    enum ConversionType { NHWC_NCHW, NCHW_NHWC };

    std::shared_ptr<ngraph::Node> transpose(ConversionType type,
                                            std::shared_ptr<ngraph::Node> input);

public:
    // static std::string sPluginType;
    // OperationsBase(const Model& model);
    // void setNgraphNodes(std::shared_ptr<NgraphNodes> nodes);
    // virtual bool validate(const Operation& op);
    // // override connectOperationToGraph in case Operation has multiple outputs
    // virtual void connectOperationToGraph(const Operation& op);
    OperationsBase(NnapiModelInfo* model, NgraphNetworkCreator* nwCreator) {
        mModelInfo = model;
        mNwCreator = nwCreator;
    }

    virtual bool createNode(const Operation& op) = 0;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif