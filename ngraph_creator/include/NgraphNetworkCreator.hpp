#include <OperationsFactory.hpp>
#include <NgraphNodes.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
    
class NgraphNetworkCreator
{
private:
    std::vector<std::shared_ptr<ngraph::Node>> mOperationOutputs;
    Model mModel;
    std::shared_ptr<NgraphNodes> mNgraphNodes;
    OperationsFactory mOpFctryInst;
    ngraph::ParameterVector mInputParams;
    std::vector<std::shared_ptr<ngraph::Node>> mResultNodes;
    void createInputParams();
public:
    NgraphNetworkCreator(const Model& model, const std::string& plugin);
    
    bool validateOperations();        
    bool initializeModel();
    
    const std::string& getNodeName(uint32_t index);
    
    std::shared_ptr<ngraph::Function> generateGraph();
    
};

}
}
}
}

