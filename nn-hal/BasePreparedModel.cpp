/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#define LOG_TAG "BasePreparedModel"
#include "BasePreparedModel.h"

#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <thread>
#include "ValidateHal.h"

#include <cutils/properties.h>

#define DISABLE_ALL_QUANT
#define LOG_TAG "BasePreparedModel"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;

static const Timing kNoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

void BasePreparedModel::deinitialize() {
    ALOGV("Entering %s", __func__);
    mModelInfo->unmapRuntimeMemPools();

    if (mNet) delete mNet;

    if (mPlugin) delete mPlugin;
    ALOGV("Exiting %s", __func__);
}

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
    // TODO: Check buffer is at least as long as size of data.
    T* data = reinterpret_cast<T*>(info.buffer);
    return data[0];
}

bool BasePreparedModel::initialize(const Model& model) {
    ALOGV("Entering %s", __func__);
    return true;
}

static Return<void> notify(const sp<V1_0::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>&, Timing) {
    return callback->notify(status);
}

static Return<void> notify(const sp<V1_2::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>& outputShapes, Timing timing) {
    return callback->notify_1_2(status, outputShapes, timing);
}

template <typename T_IExecutionCallback>
Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
                                BasePreparedModel* preparedModel,
                                const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, preparedModel->getModelInfo()->getModel())) {
        notify(callback, ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([preparedModel, request, measure, driverStart, callback] {
        asyncExecute(request, measure, preparedModel, driverStart, callback);
    }).detach();
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

template <typename T_IExecutionCallback>
void asyncExecute(const Request& request, MeasureTiming measure, BasePreparedModel* preparedModel,
                  time_point driverStart, const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();
    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }

    auto inLayerMap = ngraphNw->getInputLayerMap();
    // Verify all the request inputs are captured during graph construction
    // Can be optimized.
    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto reqIn = modelInfo->getModelInputIndex(i);
        ALOGD("Searching for input index : %d in layermap", reqIn);
        auto iter2 = std::find_if(inLayerMap.begin(), inLayerMap.end(),
                                  [&](const std::pair<uint32_t, LayerInfo>& elem) {
                                      ALOGD("Index: %d", elem.first);
                                      return (elem.first == reqIn);
                                  });
        if (iter2 == inLayerMap.end()) {
            ALOGE("Did we miss mapping some of the inputs???");
        }
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto inIndex = modelInfo->getModelInputIndex(i);
        auto srcBlob = modelInfo->getBlobFromMemoryPoolIn(request, i);

        // Get name of node from layermap
        // Ignore memory layers for sometime
        if (inLayerMap.find(inIndex) != inLayerMap.end()) {
            auto layerData = inLayerMap[inIndex];
            ALOGD("Found input index: %d layername : %s", inIndex, layerData.layerName.c_str());
            auto destBlob = plugin->getInferRequest().GetBlob(layerData.layerName);
            uint8_t* dest = destBlob->buffer().as<uint8_t*>();
            uint8_t* src = srcBlob->buffer().as<uint8_t*>();
            std::memcpy(dest, src, srcBlob->byteSize());
            writeBufferToFile(layerData.layerName, srcBlob->buffer().as<float*>(), srcBlob->size());
        } else {
            ALOGE("Failed to find layerindex in input layers");
        }
    }
    ALOGD("Run");

    plugin->infer();
    auto outLayerMap = ngraphNw->getOutputLayerMap();
    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = modelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i);

        if (outLayerMap.find(outIndex) != outLayerMap.end()) {
            auto layerData = outLayerMap[outIndex];
            ALOGD("Found output index: %d layername : %s", outIndex, layerData.layerName.c_str());
            auto srcBlob = plugin->getInferRequest().GetBlob(layerData.layerName);
            std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(), srcBlob->byteSize());
            writeBufferToFile(layerData.layerName, srcBlob->buffer().as<float*>(), srcBlob->size());

            float* a = static_cast<float*>(destPtr);
            ALOGD("########### -- %f", *a);
        } else {
            ALOGE("Failed to find layerindex in output layers");
            notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        }
    }

    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
    }

    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        returned = notify(callback, ErrorStatus::NONE, modelInfo->getOutputShapes(), timing);
    } else {
        returned = notify(callback, ErrorStatus::NONE, modelInfo->getOutputShapes(), kNoTiming);
    }
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
    ALOGV("Exiting %s", __func__);
}

static std::tuple<ErrorStatus, hidl_vec<V1_2::OutputShape>, Timing> executeSynchronouslyBase(
    const Request& request, MeasureTiming measure, BasePreparedModel* preparedModel,
    time_point driverStart) {
    ALOGV("Entering %s", __func__);
    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();
    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        ALOGE("Failed to set runtime pool info from HIDL memories");
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }

    auto inLayerMap = ngraphNw->getInputLayerMap();
    // Verify all the request inputs are captured during graph construction
    // Can be optimized.
    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto reqIn = modelInfo->getModelInputIndex(i);
        ALOGD("Searching for input index : %d in layermap", reqIn);
        auto iter2 = std::find_if(inLayerMap.begin(), inLayerMap.end(),
                                  [&](const std::pair<uint32_t, LayerInfo>& elem) {
                                      ALOGD("Index: %d", elem.first);
                                      return (elem.first == reqIn);
                                  });
        if (iter2 == inLayerMap.end()) {
            ALOGE("Did we miss mapping some of the inputs???");
        }
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto inIndex = modelInfo->getModelInputIndex(i);
        auto srcBlob = modelInfo->getBlobFromMemoryPoolIn(request, i);

        // Get name of node from layermap
        // Ignore memory layers for sometime
        if (inLayerMap.find(inIndex) != inLayerMap.end()) {
            auto layerData = inLayerMap[inIndex];
            ALOGD("Found input index: %d layername : %s", inIndex, layerData.layerName.c_str());
            auto destBlob = plugin->getInferRequest().GetBlob(layerData.layerName);
            uint8_t* dest = destBlob->buffer().as<uint8_t*>();
            uint8_t* src = srcBlob->buffer().as<uint8_t*>();
            std::memcpy(dest, src, srcBlob->byteSize());
            writeBufferToFile(layerData.layerName, srcBlob->buffer().as<float*>(), srcBlob->size());
        } else {
            ALOGE("Failed to find layerindex in input layers");
        }
    }
    ALOGD("Run");

    plugin->infer();
    auto outLayerMap = ngraphNw->getOutputLayerMap();
    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = modelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i);

        if (outLayerMap.find(outIndex) != outLayerMap.end()) {
            auto layerData = outLayerMap[outIndex];
            ALOGD("Found output index: %d layername : %s", outIndex, layerData.layerName.c_str());
            auto srcBlob = plugin->getInferRequest().GetBlob(layerData.layerName);
            std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(), srcBlob->byteSize());
            writeBufferToFile(layerData.layerName, srcBlob->buffer().as<float*>(), srcBlob->size());

            float* a = static_cast<float*>(destPtr);
            ALOGD("########### -- %f", *a);
        } else {
            ALOGE("Failed to find layerindex in output layers");
            return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
        }
    }

    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        return {ErrorStatus::NONE, modelInfo->getOutputShapes(), timing};
    }
    return {ErrorStatus::NONE, modelInfo->getOutputShapes(), kNoTiming};
    ALOGV("Exiting %s", __func__);
}

Return<void> BasePreparedModel::executeSynchronously(const Request& request, MeasureTiming measure,
                                                     executeSynchronously_cb cb) {
    ALOGV("Entering %s", __func__);
    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();
    auto [status, outputShapes, timing] =
        executeSynchronouslyBase(request, measure, this, driverStart);
    cb(status, std::move(outputShapes), timing);
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<void> BasePreparedModel::configureExecutionBurst(
    const sp<V1_2::IBurstCallback>& callback,
    const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
    const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel, configureExecutionBurst_cb cb) {
    ALOGV("Entering %s", __func__);

    cb(ErrorStatus::GENERAL_FAILURE, {});
    return Void();
}

Return<ErrorStatus> BasePreparedModel::execute(const Request& request,
                                               const sp<V1_0::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, MeasureTiming::NO, this, callback);
}

Return<ErrorStatus> BasePreparedModel::execute_1_2(const Request& request, MeasureTiming measure,
                                                   const sp<V1_2::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, measure, this, callback);
}

template std::__1::vector<unsigned int, std::__1::allocator<unsigned int> >;
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
