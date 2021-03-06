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

import ctypes
import numpy as np
import os
import pycuda.autoinit
import sys
import tensorrt as trt

from lenet5 import MODEL_DIR, ModelData
from random import randint

# ../common.py
sys.path.insert(1,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir
    )
)
import common
#指定工作空间文件夹
WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

# Path where clip plugin library will be built (check README.md)
#创建相应的插件库的路径
CLIP_PLUGIN_LIBRARY = os.path.join(
    WORKING_DIR,
    'build/libclipplugin.so'
)

# Path to which trained model will be saved (check README.md)
# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
#创建trt的logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Builds TensorRT Engine
#构建相应的引擎
def build_engine(model_path):
    #创建相应的实例
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.UffParser() as parser:
        #配置相关参数
        config.max_workspace_size = common.GiB(1)

        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_path, network)
        #构建引擎
        return builder.build_engine(network, config)

def load_test_data():
    with open(os.path.join(MODEL_DIR, "x_test.npy"), 'rb') as f:
        x_test = np.load(f)
    with open(os.path.join(MODEL_DIR, "y_test.npy"), 'rb') as f:
        y_test = np.load(f)
    return x_test, y_test

# Loads a test case into the provided pagelocked_buffer. Returns loaded test case label.
#加载测试数据到相应的主机页面锁定内存，返回对应的label
def load_normalized_test_case(pagelocked_buffer):
    x_test, y_test = load_test_data()
    num_test = len(x_test)
    case_num = randint(0, num_test-1)
    img = x_test[case_num].ravel()
    np.copyto(pagelocked_buffer, img)
    return y_test[case_num]

def main():
    # Load the shared object file containing the Clip plugin implementation.
    # By doing this, you will also register the Clip plugin with the TensorRT
    # PluginRegistry through use of the macro REGISTER_TENSORRT_PLUGIN present
    # in the plugin implementation. Refer to plugin/clipPlugin.cpp for more details.
    #判断相应的插件库是否存在
    if not os.path.isfile(CLIP_PLUGIN_LIBRARY):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load library ({}).".format(CLIP_PLUGIN_LIBRARY),
            "Please build the Clip sample plugin.",
            "For more information, see the included README.md"
        ))
        #加载相应的插件库
        #ctypes是python中用来调用外部库的函数
    ctypes.CDLL(CLIP_PLUGIN_LIBRARY)

    # Load pretrained model
    #加载模型
    model_path = os.path.join(MODEL_DIR, "trained_lenet5.uff")
    if not os.path.isfile(model_path):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load model file ({}).".format(model_path),
            "Please use 'python3 model.py' to train and save the UFF model.",
            "For more information, see README.md"
        ))

    # Build an engine and retrieve the image mean from the model.
    #构建相应的引擎文件
    #build_engine参考本文件下的实现
    with build_engine(model_path) as engine:
        #分配相应的内存缓冲区
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        #创建推理上下文
        with engine.create_execution_context() as context:
            print("\n=== Testing ===")
            #加载数据到主机内存，获取对应数据的真值laibel
            test_case = load_normalized_test_case(inputs[0].host)
            print("Loading Test Case: " + str(test_case))
            # The common do_inference function will return a list of outputs - we only have one in this case.
            #执行推理
            [pred] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print("Prediction: " + str(np.argmax(pred)))


if __name__ == "__main__":
    main()
