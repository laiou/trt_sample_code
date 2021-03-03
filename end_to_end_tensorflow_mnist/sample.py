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

# This sample uses a UFF MNIST model to create a TensorRT Inference Engine
from random import randint
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
#Logger for the Builder, ICudaEngine and Runtime .
#关于logger参考https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Logger.html?highlight=logger#tensorrt.Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#模型信息类
class ModelData(object):
    #包括模型文件路径，输入节点名称，输入节点维度，输出节点名称等信息
    MODEL_FILE = "lenet5.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"

def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.UffParser() as parser:
        config.max_workspace_size = common.GiB(1)
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_engine(network, config)

# Loads a test case into the provided pagelocked_buffer.
#加载测试数据到提供的缓冲区
#load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host)
def load_normalized_test_case(data_paths, pagelocked_buffer, case_num=randint(0, 9)):
    #获取相应的测试数据路径的列表
    [test_case_path] = common.locate_files(data_paths, [str(case_num) + ".pgm"], err_msg="MNIST image data not found. Please follow the README instructions.")
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    #ravel将图片多维数据降成一维
    img = np.array(Image.open(test_case_path)).ravel()
    #对图片数据进行归一化并且复制到指定的页面锁定内存中
    #numpy.copyto将相应的数据复制到提供的页面锁定内存中
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)
    return case_num

def main():
    #解析样本数据，获取数据路径
    #find_sample_data的具体实现参考common.py中的实现
    data_paths, _ = common.find_sample_data(description="Runs an MNIST network using a UFF model file", subfolder="mnist")
    #获取模型文件的路径
    model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "models")

    model_file = os.path.join(model_path, ModelData.MODEL_FILE)
    #创建相应的engine文件
    with build_engine(model_file) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        #allocate_buffers参考common.py
        #获取相应缓冲区地址的列表已经相应绑定的列表等
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        #create_execution_context新建一个IExecutionContext类实例
        with engine.create_execution_context() as context:
            #load_normalized_test_case的实现参考当前文件下的相关实现
            #将测试数据加载到提供的缓冲区里面
            case_num = load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            #进行相应的推理过程
            #do_inference的具体实现参考common.py
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            #得到最终的预测输出，也就是相应的后处理过程
            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))
if __name__ == '__main__':
    main()
