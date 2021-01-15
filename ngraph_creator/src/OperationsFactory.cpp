#include <OperationsFactory.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
    
OperationsFactory::OperationsFactory(const std::string& plugin) { OperationsBase::sPluginType = plugin; }
std::shared_ptr<OperationsBase> OperationsFactory::getOperation(const OperationType& type, const Model& model)
{
    auto opIter = mOperationsMap.find(type);
    if(opIter != mOperationsMap.end())
        return opIter->second;
        
    switch(type) {
        case OperationType::ADD:
            mOperationsMap[type] = std::make_shared<Add>(model);
            return mOperationsMap[type];
        case OperationType::CONCATENATION:
            mOperationsMap[type] = std::make_shared<Concat>(model);
            return mOperationsMap[type];
        default:
            return nullptr;
    }
}

}
}
}
}