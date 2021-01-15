#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

std::string OperationsBase::sPluginType;

std::shared_ptr<ngraph::Node> OperationsBase::transpose(ConversionType type, std::shared_ptr<ngraph::Node> input)
{
    ngraph::AxisVector order;
    switch(type)
    {
        case NHWC_NCHW:
            order = {0, 3, 1, 2};
            break;
        case NCHW_NHWC:
            order = {0, 2, 3, 1};
    }
    const auto order_node = ngraph::opset3::Constant::create(
        ngraph::element::i64, ngraph::Shape{order.size()}, order);
    return std::make_shared<ngraph::opset3::Transpose>(input, order_node);
}
    
//override createNodeForPlugin in case sPluginType specific implementation is required
std::shared_ptr<ngraph::Node> OperationsBase::createNodeForPlugin(const Operation& op, const std::vector<std::shared_ptr<ngraph::Node>>& nodes)
{ return createNode( op, nodes); }

OperationsBase::OperationsBase(const Model& model) : mModel(model) {}

bool OperationsBase::validate(const Operation& op)
{ return true; }

}
}
}
}