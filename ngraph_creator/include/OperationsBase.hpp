#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <Driver.h>
#include <android/log.h>
#include <log/log.h>
#include <Temp.h> //TODO: Remove this once NNAPI_Utils is ready

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class OperationsBase
{
protected:
    Model mModel;
    enum ConversionType
    {
        NHWC_NCHW,
        NCHW_NHWC
    };
    std::shared_ptr<ngraph::Node> transpose(ConversionType type, std::shared_ptr<ngraph::Node> input);
    virtual std::shared_ptr<ngraph::Node> createNode(const Operation& op, const std::vector<std::shared_ptr<ngraph::Node>>& nodes) = 0;
    
public:
    static std::string sPluginType;
    OperationsBase(const Model& model);
    virtual bool validate(const Operation& op);
    
    //override createNodeForPlugin in case sPluginType specific implementation is required
    virtual std::shared_ptr<ngraph::Node> createNodeForPlugin(const Operation& op, const std::vector<std::shared_ptr<ngraph::Node>>& nodes);

};


}
}
}
}