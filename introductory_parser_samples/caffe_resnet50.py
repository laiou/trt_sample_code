#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This sample uses a Caffe ResNet50 Model to create a TensorRT Inference Engine
import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
#模型信息类，包含模型路径，输入的维度，以及输出结点的名称
class ModelData(object):
    MODEL_PATH = "ResNet50_fp32.caffemodel"
    DEPLOY_PATH = "ResNet50_N2.prototxt"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "prob"
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
#创建tensorrt的logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Allocate host and device buffers, and create a stream.
#分配相应的内存缓冲区
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    #给相应的输入和输出数据分配主机内存
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    #分配相应的设备内存
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

# The Caffe path is used for Caffe2 models.
#构建相应的引擎
def build_engine_caffe(model_file, deploy_file):
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    #创建相关实例
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
        # Workspace size is the maximum amount of memory available to the builder while building an engine.
        # It should generally be set as high as possible.
        #设置相应的参数
        config.max_workspace_size = common.GiB(1)
        # Load the Caffe model and parse it in order to populate the TensorRT network.
        # This function returns an object that we can query to find tensors by name.
        #使用caffe解析器解析模型
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        # For Caffe, we need to manually mark the output of the network.
        # Since we know the name of the output tensor, we can find it in model_tensors.
        #标记模型输出结点
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        #构建相应的引擎文件
        return builder.build_engine(network, config)
#转换输入数据
def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

def main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    #解析样本数据
    _, data_files = common.find_sample_data(description="Runs a ResNet50 network with a TensorRT inference engine.", subfolder="resnet50", find_files=["binoculars.jpeg", "reflex_camera.jpeg", "tabby_tiger_cat.jpg", ModelData.MODEL_PATH, ModelData.DEPLOY_PATH, "class_labels.txt"])
    # Get test images, models and labels.
    #获取测试的图片数据
    test_images = data_files[0:3]
    #相应的模型文件和label文件
    caffe_model_file, caffe_deploy_file, labels_file = data_files[3:]
    #读取label文件
    labels = open(labels_file, 'r').read().split('\n')

    # Build a TensorRT engine.
    #构建相应的tensorrt引擎
    #build_engine_caffe参考本文件下的实现
    with build_engine_caffe(caffe_model_file, caffe_deploy_file) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        # allocate_buffers参考本文件下的实现
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        # Contexts are used to perform inference.
        #创建相关的上下文实例
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            #随机选择测试数据
            test_image = random.choice(test_images)
            #load_normalized_test_case参考本文件下的实现
            #加载输入数据，并完成相应的维度变换
            test_case = load_normalized_test_case(test_image, h_input)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            #进行具体的推理
            #do_inference参考本文件下的实现
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            #获取最终的预测输出
            pred = labels[np.argmax(h_output)]
            if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
                print("Correctly recognized " + test_case + " as " + pred)
            else:
                print("Incorrectly recognized " + test_case + " as " + pred)

if __name__ == '__main__':
    main()
