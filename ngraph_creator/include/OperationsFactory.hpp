#include <Add.hpp>
#include <Concat.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
    
class OperationsFactory
{
private:
    std::map<OperationType, std::shared_ptr<OperationsBase>> mOperationsMap;
public:
    OperationsFactory(const std::string& plugin);
    std::shared_ptr<OperationsBase> getOperation(const OperationType& type, const Model& model);
};

}
}
}
}