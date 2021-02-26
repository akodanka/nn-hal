#define LOG_TAG "GnaPreparedModel"

#include "GnaPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include "OperationsFactory.hpp"
#include "ValidateHal.h"
#include "utils.h"

using namespace android::nn;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

void GnaPreparedModel::deinitialize() {
    ALOGV("Entering %s", __func__);
    mModelInfo->unmapRuntimeMemPools();

    if (mNet) delete mNet;

    if (mPlugin) delete mPlugin;
    ALOGV("Exiting %s", __func__);
}

bool GnaPreparedModel::initialize(const Model &model) {
    ALOGV("Entering %s", __func__);
    // mPlugin = new DevicePlugin();
    if (!mModelInfo->initRuntimeInfo()) {
        ALOGE("Failed to initialize Model runtime parameters!!");
        return false;
    }
    mNgc = std::make_shared<NgraphNetworkCreator>(mModelInfo.get(), mTargetDevice);

    auto vecOperations = mModelInfo->getOperations();
    for (auto op : vecOperations) {
        if (!OperationsFactory::isOperationSupported(op, mModelInfo.get())) {
            ALOGE("Unsupported operation");
            return false;
        }
    }

    ALOGI("Generating IR Graph");
    mNet = mNgc->generateIRGraph();
    // mNet.serialize("/data/vendor/neuralnetworks/ngraph_ir.xml",
    //                      "/data/vendor/neuralnetworks/ngraph_ir.bin");
    mPlugin = new IENetwork(mNet);
    mPlugin->loadNetwork();
    // ALOGI("initialize ExecuteNetwork for device %s", mTargetDevice.c_str());
    // mEnginePtr = new ExecuteNetwork(mNet, mTargetDevice);
    // mEnginePtr->prepareInput();
    // mEnginePtr->loadNetwork(mNet);
    ALOGV("Exiting %s", __func__);
    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
