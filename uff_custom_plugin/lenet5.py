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

# This file contains functions for training a TensorFlow model
import os
import tensorrt as trt
#指定相应的工作文件加
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
#以及相应的模型文件夹
MODEL_DIR = os.path.join(
    WORKING_DIR,
    'models'
)
#指定相应的数据信息
# MNIST dataset metadata
MNIST_IMAGE_SIZE = 28
MNIST_CHANNELS = 1
MNIST_CLASSES = 10
#模型信息类，输入输出节点的名称，维度等信息
class ModelData(object):
    INPUT_NAME = "InputLayer"
    INPUT_SHAPE = (MNIST_CHANNELS, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    RELU6_NAME = "ReLU6"
    OUTPUT_NAME = "OutputLayer/Softmax"
    OUTPUT_SHAPE = (MNIST_IMAGE_SIZE, )
    DATA_TYPE = trt.float32
