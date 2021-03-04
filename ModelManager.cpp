#include "ModelManager.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

bool NnapiModelInfo::initializeRunTimeOperandInfo() {
    // initialize runtime operand info from model.
    const size_t count = mModel.operands.size();
    if (!count) {
        ALOGE("NNERR:Operand Count is 0");
        return false;
    }
    mOperands.resize(count);
    mOutputShapes.resize(mModel.outputIndexes.size());

    // Start by setting the runtime info to what's in the model.
    for (size_t i = 0; i < count; i++) {
        const Operand& from = mModel.operands[i];
        RunTimeOperandInfo& to = mOperands[i];
        to.dimensions.resize(from.dimensions.size());
        for (size_t j = 0; j < from.dimensions.size(); j++) {
            to.dimensions[j] = from.dimensions[j];
        }

        to.scale = from.scale;
        switch (from.type) {
            case OperandType::TENSOR_FLOAT32:
            case OperandType::FLOAT32:
                to.type = OperandType::TENSOR_FLOAT32;
                ALOGD("OperandType = %d\n", from.type);
                break;
            case OperandType::INT32:
            case OperandType::UINT32:
                nnAssert(to.scale == 0);
                FALLTHROUGH_INTENDED;
            case OperandType::TENSOR_INT32:
                to.type = from.type;
                break;
            case OperandType::TENSOR_QUANT8_ASYMM:
                ALOGE("OperandType::TENSOR_QUANT8_ASYMM is not supported");
                break;
            default:
                ALOGE("wrong operand type %d", from.type);
                return false;
        }

        to.length = from.location.length;
        to.lifetime = from.lifetime;
        to.zeroPoint = from.zeroPoint;

        switch (from.lifetime) {
            case OperandLifeTime::TEMPORARY_VARIABLE:
                to.buffer = nullptr;
                to.length = sizeOfData(to.type, to.dimensions);
                to.numberOfUsesLeft = from.numberOfConsumers;
                break;
            case OperandLifeTime::CONSTANT_COPY:
                to.buffer = const_cast<uint8_t*>(&mModel.operandValues[from.location.offset]);
                to.numberOfUsesLeft = 0;
                break;
            case OperandLifeTime::CONSTANT_REFERENCE: {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < mPoolInfos.size());
                auto& r = mPoolInfos[poolIndex];
                to.buffer = r.buffer + from.location.offset;
                to.numberOfUsesLeft = 0;
                break;
            }
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                to.buffer = nullptr;
                to.numberOfUsesLeft = 0;
                break;
            default:
                return false;
                break;
        }
    }

    for (uint32_t i = 0; i < mModel.outputIndexes.size(); i++) {
        const uint32_t operandIndex = mModel.outputIndexes[i];
        const RunTimeOperandInfo& from = mOperands[operandIndex];
        mOutputShapes[i].dimensions = from.dimensions;
        mOutputShapes[i].isSufficient = true;
    }

    return true;
}

// TODO: Move it to Utils class
template <typename T>
T NnapiModelInfo::GetConstFromBuffer(const uint8_t* buf, uint32_t len) {
    ALOGD("buf: %p, len: %d", buf, len);
    if (len != sizeof(T)) {
        ALOGE("fix me: typeid(T).name() should be %d bytes", sizeof(T));
        // fix me if buffer is of type float and if float and OperandLifeTime::CONSTANT_REFERENCE
        nnAssert(false);
    }
    return *(T*)(buf);
}

const uint8_t* NnapiModelInfo::GetOperandMemory(int index, uint32_t& lenOut) {
    ALOGD("%s", __func__);
    const auto op = mModel.operands[index];
    lenOut = op.location.length;
    if (op.lifetime == OperandLifeTime::CONSTANT_COPY) {
        ALOGD("CONST_COPY");
        if (op.location.poolIndex != 0) {
            ALOGE("CONSTANT_COPY expects poolIndex to be 0");
            nnAssert(false);
        }
        ALOGD("operand lifetime OperandLifeTime::CONSTANT_COPY");
        return (const_cast<uint8_t*>(&mModel.operandValues[op.location.offset]));
    } else if (op.lifetime == OperandLifeTime::CONSTANT_REFERENCE) {
        ALOGD("operand lifetime OperandLifeTime::CONSTANT_REFERENCE");
        auto poolIndex = op.location.poolIndex;
        auto& r = mPoolInfos[poolIndex];
        return (const_cast<uint8_t*>(r.buffer + op.location.offset));
    } else if (op.lifetime == OperandLifeTime::TEMPORARY_VARIABLE ||
               op.lifetime == OperandLifeTime::MODEL_INPUT ||
               op.lifetime == OperandLifeTime::MODEL_OUTPUT ||
               op.lifetime == OperandLifeTime::NO_VALUE) {
        ALOGD(
            "operand lifetime "
            "OperandLifeTime::MODEL_INPUT||MODEL_OUTPUT||NO_VALUE||TEMPORARY_VARIABLE");
        lenOut = sizeOfData(op.type, op.dimensions);
        return nullptr;
    }
    ALOGE("operand is expected to be const, but lifetime is %d", op.lifetime);
    nnAssert(false);  // temp fix since some time const operand set as TEMPORARY_VARIABLE
    return nullptr;
}

Blob::Ptr NnapiModelInfo::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                                uint32_t& len) {
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
            ALOGD("Create input blob !!!!");
            vec<unsigned int> order;
            InferenceEngine::Layout layout;
            if (op.dimensions.size() == 4) {
                order = {0, 3, 1, 2};  // nhwc -> nchw
                layout = InferenceEngine::Layout::NCHW;
            } else if (op.dimensions.size() == 2) {
                order = {0, 1};
                layout = InferenceEngine::Layout::NC;
            } else {
                order = {0};  //(op.dimensions.size() < 2)
                layout = InferenceEngine::Layout::C;
            }

            auto inputDims = toDims(op.dimensions);
            InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32,
                                           permuteDims(inputDims, order), layout);

            if (buf == nullptr) {
                ALOGD("MODEL_INPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                if (inputDims.size() != 4) {
                    InferenceEngine::TBlob<float>::Ptr blob =
                        std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf);
                    return blob;
                } else {
                    InferenceEngine::TBlob<float>::Ptr blob =
                        std::make_shared<InferenceEngine::TBlob<float>>(td);
                    blob->allocate();

                    auto dims_nhwc = inputDims;  // toDims(op.dimensions);
                    size_t batch = dims_nhwc[0];
                    size_t in_depth = dims_nhwc[3];  // channels
                    size_t height = dims_nhwc[1];
                    size_t width = dims_nhwc[2];
                    size_t offset = 0;  // blob->size() == o*i*h*w and simlar to nchw memory layout
                    const float* input = reinterpret_cast<const float*>(buf);  // OHWI memory layout

                    // convert NHWC -> NCHW

                    for (size_t b = 0; b < batch; b++) {
                        for (size_t i = 0; i < in_depth; i++) {
                            for (size_t h = 0; h < height; h++) {
                                for (size_t w = 0; w < width; w++) {
                                    size_t offset_nhwc = b * height * width * in_depth +
                                                         h * width * in_depth + w * in_depth +
                                                         i;  // similar to NHWC memory layout
                                    blob->buffer().as<float*>()[offset++] = input[offset_nhwc];
                                }
                            }
                        }
                    }

                    return blob;
                }
            }
        } else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            ALOGD("Create output blob !!!!");
            vec<unsigned int> order;
            InferenceEngine::Layout layout;
            if (op.dimensions.size() == 4) {
                // order = {0,3,1,2};  //nhwc -> nchw
                layout = InferenceEngine::Layout::NHWC;
            } else if (op.dimensions.size() == 2) {
                // order = {0, 1};
                layout = InferenceEngine::Layout::NC;
            } else if (op.dimensions.size() == 3) {
                // order = {0, 1, 2, 3};  // nhwc -> nchw
                layout = InferenceEngine::Layout::CHW;
                ALOGI("Anoob : GetInOutOperandAsBlob output already transposed to NHWC");
            } else {
                // order = {0}; //(op.dimensions.size() < 2)
                layout = InferenceEngine::Layout::C;
            }

            InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions),
                                           layout);  // nhwc
            if (buf == nullptr) {
                VLOG(L1, "MODEL_OUTPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    InferenceEngine::make_shared_blob<float>(td, (float*)buf);
                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        VLOG(L1, "check if const tensors of type IN32 supported");
        // nnAssert(true);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t*)buf, len);
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

IRBlob::Ptr NnapiModelInfo::GetConstOperandAsTensor(int operand_idx, int operation_idx) {
    dumpOperand(operand_idx, mModel);
    const auto op = mModel.operands[operand_idx];
    uint32_t len;

    const uint8_t* buf = GetOperandMemory(operand_idx, len);
    VLOG(L1, "NnapiModelInfo:: operand_index: %d, operation_index :%d,len: %d, buf: %p",
         operand_idx, operation_idx, len, buf);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        InferenceEngine::Layout layout;
        if (op.dimensions.size() == 4) {
            order = {0, 3, 1, 2};                    // nhwc -> nchw
            layout = InferenceEngine::Layout::OIHW;  // weights layout
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = InferenceEngine::Layout::NC;
        } else {
            order = {0};  //(op.dimensions.size() < 2)
            layout = InferenceEngine::Layout::C;
        }
        auto inputDims = toDims(op.dimensions);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32,
                                       permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_ohwi = inputDims;  // toDims(op.dimensions);
                size_t out_depth = dims_ohwi[0];
                size_t in_depth = dims_ohwi[3];
                size_t height = dims_ohwi[1];
                size_t width = dims_ohwi[2];
                size_t offset = 0;  // blob->size() == o*i*h*w and simlar to nchw memory layout
                const float* inputFilter =
                    reinterpret_cast<const float*>(buf);  // OHWI memory layout

                for (size_t o = 0; o < out_depth; o++) {
                    for (size_t i = 0; i < in_depth; i++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                size_t offset_ohwi = o * height * width * in_depth +
                                                     h * width * in_depth + w * in_depth +
                                                     i;  // similar to NHWC memory layout
                                blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                            }
                        }
                    }
                }

                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        VLOG(L1, "check if const tensors of type IN32 supported");
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

// Redundant.. Remove the code
IRBlob::Ptr NnapiModelInfo::GetConstWeightsOperandAsTensor(uint32_t index) {
    dumpOperand(index, mModel);
    const auto op = mModel.operands[index];
    uint32_t len;
    const uint8_t* buf = GetOperandMemory(index, len);
    VLOG(L1, "NnapiModelInfo:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        InferenceEngine::Layout layout;
        if (op.dimensions.size() == 4) {
            // order = {0,3,1,2};  //nhwc -> nchw
            order = {3, 0, 1, 2};                    // IHWO -> OIHW for depth conv
            layout = InferenceEngine::Layout::OIHW;  // weights layout
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = InferenceEngine::Layout::NC;
        } else {
            order = {0};  //(op.dimensions.size() < 2)
            layout = InferenceEngine::Layout::C;
        }
        auto inputDims = toDims(op.dimensions);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32,
                                       permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_ohwi = inputDims;  // toDims(op.dimensions);
                size_t out_depth = dims_ohwi[0];
                size_t in_depth = dims_ohwi[3];
                size_t height = dims_ohwi[1];
                size_t width = dims_ohwi[2];
                size_t offset = 0;  // blob->size() == o*i*h*w and simlar to nchw memory layout
                const float* inputFilter =
                    reinterpret_cast<const float*>(buf);  // OHWI memory layout

                // convert OHWI -> OIHW

                // for depth conv need reorder as IOHW since for tflite O is always 1 and IE expects
                // reorder to [in_channels, depth_multiplier, filter_height, filter_width]
                for (size_t i = 0; i < in_depth; i++) {
                    for (size_t o = 0; o < out_depth; o++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                size_t offset_ohwi = o * height * width * in_depth +
                                                     h * width * in_depth + w * in_depth +
                                                     i;  // similar to NHWC memory layout
                                blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                            }
                        }
                    }
                }

                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        VLOG(L1, "check if const tensors of type IN32 supported");
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions),
                                       InferenceEngine::Layout::ANY);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

bool NnapiModelInfo::setRunTimePoolInfosFromHidlMemories(const hidl_vec<hidl_memory>& pools) {
    ALOGD("Number of pools: %d", pools.size());
    mRequestPoolInfos.resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = mRequestPoolInfos[i];
        if (!poolInfo.set(pools[i])) {
            ALOGE("Could not map memory pool !!!");
            return false;
        }
    }
    return true;
}

Blob::Ptr NnapiModelInfo::getBlobFromMemoryPoolIn(const Request& request, uint32_t index) {
    RunTimeOperandInfo& operand = mOperands[mModel.inputIndexes[index]];
    const V1_0::RequestArgument& arg = request.inputs[index];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRequestPoolInfos.size());
    auto& r = mRequestPoolInfos[poolIndex];

    if (arg.dimensions.size() > 0) {
        // It's the responsibility of the caller to validate that
        // from.dimensions only modifies the dimensions that were
        // unspecified in the model.  That's the case in SampleDriver.cpp
        // with the call to validateRequest().
        operand.dimensions = arg.dimensions;
    }

    operand.buffer = r.buffer + arg.location.offset;
    operand.length = arg.location.length;
    ALOGI("%s Operand length:%d pointer:%p offset:%d pool index: %d", __func__, operand.length,
          (r.buffer + arg.location.offset), arg.location.offset, poolIndex);
    return GetInOutOperandAsBlob(operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                 operand.length);
}

void* NnapiModelInfo::getBlobFromMemoryPoolOut(const Request& request, uint32_t index) {
    RunTimeOperandInfo& operand = mOperands[mModel.outputIndexes[index]];
    const V1_0::RequestArgument& arg = request.outputs[index];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRequestPoolInfos.size());
    auto& r = mRequestPoolInfos[poolIndex];

    ALOGD("%s lifetime:%d location offset:%d length:%d pool index:%d", __func__, operand.lifetime,
          arg.location.offset, arg.location.length, poolIndex);

    if (arg.dimensions.size() > 0) {
        // It's the responsibility of the caller to validate that
        // from.dimensions only modifies the dimensions that were
        // unspecified in the model.  That's the case in SampleDriver.cpp
        // with the call to validateRequest().
        operand.dimensions = arg.dimensions;
    }

    operand.buffer = r.buffer + arg.location.offset;
    operand.length = arg.location.length;
    ALOGI("%s Operand length:%d pointer:%p", __func__, operand.length,
          (r.buffer + arg.location.offset));
    return (r.buffer + arg.location.offset);
}

template int NnapiModelInfo::GetConstOperand<int>(unsigned int);
template unsigned int NnapiModelInfo::GetConstOperand<unsigned int>(unsigned int);
template int NnapiModelInfo::GetConstFromBuffer<int>(unsigned char const*, unsigned int);

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android