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

# This sample uses an MNIST PyTorch model to create a TensorRT Inference Engine
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

import model

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

# Populate the TRT network, injecting some dummy weights
#用相应网络的权重填充tensorrt的network
def populate_network_with_some_dummy_weights(network, weights):
    # Configure the network layers based on the weights provided.
    #给tensorrt的network添加相应的输出层
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # Set dummy weights for the kernel and bias weights in the conv1 layer. We
    # will refit the engine with the actual weights later.
    #给相应的卷积层设置初始化的权值和偏置
    conv1_w = np.zeros((20,5,5), dtype=np.float32)
    conv1_b = np.zeros(20, dtype=np.float32)
    #将相应的层次添加到tensorrt的network中
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
    #设定层次的名称和步长
    conv1.name = "conv_1"
    conv1.stride = (1, 1)
    #同样的给tensorrt的network添加池化层
    pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)
    #中间层次直接用相应的weights中的值来填充
    conv2_w = weights['conv2.weight'].numpy()
    conv2_b = weights['conv2.bias'].numpy()
    #添加相应的层次到network，同时注意当前层次的输入的获取
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
    conv2.stride = (1, 1)

    pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride = (2, 2)

    fc1_w = weights['fc1.weight'].numpy()
    fc1_b = weights['fc1.bias'].numpy()
    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    fc2_w = weights['fc2.weight'].numpy()
    fc2_b = weights['fc2.bias'].numpy()
    fc2 = network.add_fully_connected(relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b)

    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    #最后通过mark_output将一个张量标记为输出
    network.mark_output(tensor=fc2.get_output(0))

# Build a TRT engine, but leave out some weights
##构建一个tensorrt的引擎
def build_engine_with_some_missing_weights(weights):
    # For more information on TRT basics, refer to the introductory samples.
    #创建builder和network实例
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
         #设置最大工作空间的大小，GIB的具体实现参考common.py
        builder.max_workspace_size = common.GiB(1)
        # Set the refit flag in the builder
         #在builder中设置refit标志位
        builder.refittable = True
        # Populate the network using weights from the PyTorch model.
        #利用相应模型的权重填充tensorrt网络
        #populate_network_with_some_dummy_weights的具体实现参考本文件的相应实现
        populate_network_with_some_dummy_weights(network, weights)
        # Build and return an engine.
        #建立相应的引擎
        #Builds an ICudaEngine from a INetworkDefinition 
        return builder.build_cuda_engine(network)

# Copy an image to the pagelocked input buffer
#load_img_to_input_buffer(test_img, pagelocked_buffer=inputs[0].host)
#将相应的图像数据复制到页面锁定的内存缓冲区
def load_img_to_input_buffer(img, pagelocked_buffer):
    np.copyto(pagelocked_buffer, img)

# Get the accuracy on the test set using TensorRT
#获取相应的测试准确率
#get_trt_test_accuracy(engine, inputs, outputs, bindings, stream, mnist_model)
def get_trt_test_accuracy(engine, inputs, outputs, bindings, stream, mnist_model):
    #创建一个IExecutionContext上下文实例
    with engine.create_execution_context() as context:
        #用于相应的数据统计
        correct = 0
        total = 0
        # Run inference on every sample.
        # Technically this could be batched, however this only comprises a fraction of total
        # time spent in the test.
        #循环遍历每一个测试数据
        #get_all_test_samples的具体实现参考model.py
        for test_img, test_name in mnist_model.get_all_test_samples():
            #load_img_to_input_buffer的具体实现参考本文件下的实现
            #加载图片数据到页面锁定的内存缓冲区
            load_img_to_input_buffer(test_img, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            #进行相应的推理，do_inference的具体实现参考common.py
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            #得到相应的输出，也就是后处理的过程
            pred = np.argmax(output)
            correct += (test_name == pred)
            total += 1

        accuracy = float(correct)/total
        print("Got {} correct predictions out of {} ({:.1f}%)".format(correct, total, 100 * accuracy))

        return accuracy

def main():
    #add_help参考common.py中的实现，实际上是一个命令行参数解析器
    common.add_help(description="Runs an MNIST network using a PyTorch model")
    # Train the PyTorch model
    #训练相应的模型
    #创建一个模型
    mnist_model = model.MnistModel()
    #进行训练
    mnist_model.learn()
    #提取相应的权重
    weights = mnist_model.get_weights()
    # Do inference with TensorRT.
    #在tensorrt中进行相应的推理
    #build_engine_with_some_missing_weights参考本文件中的具体实现
    with build_engine_with_some_missing_weights(weights) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        #allocate_buffers的具体实现参考common.py
        #分配相应的缓冲区，返回输入输出数据缓冲区指列表和相应的绑定等列表
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        print("Accuracy Before Engine Refit")
        #进行相应的推理并计算准确率
        get_trt_test_accuracy(engine, inputs, outputs, bindings, stream, mnist_model)

        # Refit the engine with the actual trained weights for the conv_1 layer.
        #用训练过的第一个卷积层的权值重新填充引擎
        #Refitter用来更新引擎中的权重，具体参考https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Refitter.html?highlight=refitter#tensorrt.Refitter
        with trt.Refitter(engine, TRT_LOGGER) as refitter:
            # To get a list of all refittable layers and associated weightRoles
            # in the network, use refitter.get_all()
            # Set the actual weights for the conv_1 layer. Since it consists of
            # kernel weights and bias weights, set each of them by specifying
            # the WeightsRole.
            #set_weights用于给指定的层次指定新的权值
            #具体参考https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Refitter.html?highlight=set_weights#tensorrt.Refitter.set_weights
            refitter.set_weights("conv_1", trt.WeightsRole.KERNEL,
                    weights['conv1.weight'].numpy())
            refitter.set_weights("conv_1", trt.WeightsRole.BIAS,
                    weights['conv1.bias'].numpy())
            # Get description of missing weights. This should return empty
            # lists in this case.
            #get_missing用来获取相应丢失权重的描述
            [missingLayers, weightRoles] = refitter.get_missing()
            #判断是否存在丢失权重的层次
            assert len(missingLayers) == 0, "Refitter found missing weights. Call set_weights() for all missing weights"
            # Refit the engine with the new weights. This will return True if
            # the refit operation succeeded.
            #refit_cuda_engine用来更新相关的引擎，如果成功返回true
            assert refitter.refit_cuda_engine()
        #get_latest_test_set_accuracy的具体实现参考model.py中的实现
        #用来获取最后一次训练得到的准确率
        expected_correct_predictions = mnist_model.get_latest_test_set_accuracy()
        print("Accuracy After Engine Refit (expecting {:.1f}% correct predictions)".format(100 * expected_correct_predictions))
        #获取相应的tensorrt的推理准确率
        assert get_trt_test_accuracy(engine, inputs, outputs, bindings, stream, mnist_model) >= expected_correct_predictions

if __name__ == '__main__':
    main()
