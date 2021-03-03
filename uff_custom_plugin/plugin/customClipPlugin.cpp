/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "customClipPlugin.h"
#include "NvInfer.h"
#include "clipKernel.h"

#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

// Clip plugin specific constants
//命名空间
namespace
{
const char* CLIP_PLUGIN_VERSION{"1"};
const char* CLIP_PLUGIN_NAME{"CustomClipPlugin"};
} // namespace

// Static class fields initialization
//静态类字段初始化
//具体的声明参考customClipPlugin.h
PluginFieldCollection ClipPluginCreator::mFC{};
//vector是一个存放各种动态类型的数组
std::vector<PluginField> ClipPluginCreator::mPluginAttributes;
//对定义好的插件通过REGISTER_TENSORRT_PLUGIN进行注册
REGISTER_TENSORRT_PLUGIN(ClipPluginCreator);

// Helper function for serializing plugin
//序列化插件
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
//反序列化插件的函数
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

ClipPlugin::ClipPlugin(const std::string name, float clipMin, float clipMax)
    : mLayerName(name)
    , mClipMin(clipMin)
    , mClipMax(clipMax)
{
}

ClipPlugin::ClipPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    //反序列化插件,读取相应的值
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    mClipMin = readFromBuffer<float>(d);
    mClipMax = readFromBuffer<float>(d);
    //判断相关的指针是否正确
    assert(d == (a + length));
}

const char* ClipPlugin::getPluginType() const noexcept
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPlugin::getPluginVersion() const noexcept
{
    return CLIP_PLUGIN_VERSION;
}

int ClipPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims ClipPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);

    // Clipping doesn't change input dimension, so output Dims will be the same as input Dims
    return *inputs;
}

int ClipPlugin::initialize() noexcept
{
    return 0;
}
//执行插件操作的队列
int ClipPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) noexcept
{
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    //clipInference的具体实现参拷clipKernel.cu
    status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output);

    return status;
}

size_t ClipPlugin::getSerializationSize() const noexcept
{
    return 2 * sizeof(float);
}
//序列化插件
void ClipPlugin::serialize(void* buffer) const noexcept 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mClipMin);
    writeToBuffer(d, mClipMax);

    assert(d == a + getSerializationSize());
}

void ClipPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int) noexcept
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kLINEAR);

    // Fetch volume for future enqueue() operations
    //获取相应输入数据的尺度
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++) {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
}

bool ClipPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kLINEAR)
        return true;
    else
        return false;
}

void ClipPlugin::terminate() noexcept {}

void ClipPlugin::destroy() noexcept {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* ClipPlugin::clone() const noexcept
{
    auto plugin = new ClipPlugin(mLayerName, mClipMin, mClipMax);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void ClipPlugin::setPluginNamespace(const char* libNamespace) noexcept 
{
    mNamespace = libNamespace;
}

const char* ClipPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

ClipPluginCreator::ClipPluginCreator()
{
    // Describe ClipPlugin's required PluginField arguments
    //emplace_back是就地构造，创建相应的参数
    mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    //对相应的mFC进行填充
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ClipPluginCreator::getPluginName() const noexcept
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPluginCreator::getPluginVersion() const noexcept
{
    return CLIP_PLUGIN_VERSION;
}
//获取相应mFC地址
const PluginFieldCollection* ClipPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
//创建一个插件
IPluginV2* ClipPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    float clipMin, clipMax;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    //读取相应的参数字段
    assert(fc->nbFields == 2);
    for (int i = 0; i < fc->nbFields; i++){
        //将相应的参数进行赋值
        if (strcmp(fields[i].name, "clipMin") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMin = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "clipMax") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMax = *(static_cast<const float*>(fields[i].data));
        }
    }
    //返回一个新的插件类
    return new ClipPlugin(name, clipMin, clipMax);
}
//创建一个插件,反序列化相应的插件
IPluginV2* ClipPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    //返回一个新的ClipPlugin
    return new ClipPlugin(name, serialData, serialLength);
}
//设置插件的命名空间
void ClipPluginCreator::setPluginNamespace(const char* libNamespace) noexcept 
{
    mNamespace = libNamespace;
}
//获取相应插件的命名空间
const char* ClipPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
