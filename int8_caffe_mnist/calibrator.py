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

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

# Returns a numpy buffer of shape (num_images, 1, 28, 28)
#加载相应的数据
def load_mnist_data(filepath):
    #读取文件
    with open(filepath, "rb") as f:
        #formstring实现从字符到Ascii码的转换
        raw_buf = np.fromstring(f.read(), dtype=np.uint8)
    # Make sure the magic number is what we expect
    #确定相应的数值是否正确
    assert raw_buf[0:4].view(">i4")[0] == 2051
    #提取相应的图片数量
    num_images = raw_buf[4:8].view(">i4")[0]
    #设置图片的通道数
    image_c = 1
    #以及图片的w,h
    image_h = raw_buf[8:12].view(">i4")[0]
    image_w = raw_buf[12:16].view(">i4")[0]
    # Need to scale all values to the range of [0, 1]
    #对相应的图片数据值进行归一化和相应的reshape
    #np.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组
    #让程序运行得更快
    return np.ascontiguousarray((raw_buf[16:] / 255.0).astype(np.float32).reshape(num_images, image_c, image_h, image_w))

# Returns a numpy buffer of shape (num_images)
#读取相应的label
def load_mnist_labels(filepath):
    #读取文件
    with open(filepath, "rb") as f:
        #还是将字符串转换成ascii码
        raw_buf = np.fromstring(f.read(), dtype=np.uint8)
    # Make sure the magic number is what we expect
    assert raw_buf[0:4].view(">i4")[0] == 2049
    num_labels = raw_buf[4:8].view(">i4")[0]
    #和前面读取图片数据类似
    return np.ascontiguousarray(raw_buf[8:].astype(np.int32).reshape(num_labels))
#int8的校准类
class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=64):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        #显示调用相应的父类函数
        trt.IInt8EntropyCalibrator2.__init__(self)
        #给相应的参数赋值
        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_mnist_data(training_data)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        #分配相应的设备内存
        #mem_alloc返回A DeviceAllocation 对象，表示设备内存的线性部分。
        #参考https://www.osgeo.cn/pycuda/driver.html?highlight=mem_alloc#pycuda.driver.mem_alloc
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    #获取一个batch—size的数据
    def get_batch(self, names):、
        #判断数据索引是否正确
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None
        #定位到当前需要提取的数据位置
        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))
        #提取相应的数据并降维成一维
        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        # 将相应的batch-size的数据从主机复制到设备内存
        cuda.memcpy_htod(self.device_input, batch)
        #更新相应的数据位置
        self.current_index += self.batch_size
        #返回设备内存的指针
        return [self.device_input]

    #读取校准缓存
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        #如果存在现成的校准缓存，直接读取
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
    #写校准缓存文件
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
