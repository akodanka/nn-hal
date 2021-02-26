#pragma once

#include <Driver.h>
#include <log/log.h>
#include <ngraph/shape.hpp>

#define OP_INPUT_IDX_CONV 0
static const std::string INVALID_STRING("Invalid Node");

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

struct LayerInfo {
    std::string layerName;
    bool memoryLayer;

    LayerInfo(std::string layer = "", bool mem = false) : layerName(layer), memoryLayer(mem) {}
    LayerInfo(const LayerInfo& layer) {
        layerName = layer.layerName;
        memoryLayer = layer.memoryLayer;
    }
    LayerInfo& operator=(const LayerInfo& layer) {
        layerName = layer.layerName;
        memoryLayer = layer.memoryLayer;

        return *this;
    }
};

static ngraph::Shape toNgraphShape(const std::vector<uint32_t>& dimensions) {
    ngraph::Shape shapeVec;
    for (auto i=0; i < dimensions.size(); i++) {
        shapeVec.push_back(static_cast<size_t>(dimensions[i]));
    }
    return shapeVec;
}

static void calculateExplicitPadding(int32_t in_size, int32_t stride, int32_t filter_size,
                              int32_t padding_implicit, int32_t* padding_head,
                              int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;

    if (padding_implicit == 1) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android