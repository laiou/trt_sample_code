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

from itertools import chain
import argparse
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import tensorrt as trt

try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30


def add_help(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()

#find_sample_data(description="Runs an MNIST network using a UFF model file", subfolder="mnist")
#解析样本数据
def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[], err_msg=""):
    '''
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    '''

    # Standard command-line arguments for all samples.
    #获取相关数据的根目录
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    #解析命令行参数
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory, and any additional data directories.", action="append", default=[kDEFAULT_DATA_ROOT])
    args, _ = parser.parse_known_args()
    #获取数据路径的函数
    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            if data_dir != kDEFAULT_DATA_ROOT:
                print("WARNING: " + data_path + " does not exist. Trying " + data_dir + " instead.")
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)) and data_dir != kDEFAULT_DATA_ROOT:
            print("WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(data_path))
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files, err_msg)
#在指定的文件夹里面定位相应的文件
def locate_files(data_paths, filenames, err_msg=""):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    #统计相应的文件个数，[None]表示列表里面有一个None元素，[]表示一个空列表
    found_files = [None] * len(filenames)
    #循环遍历data_paths中的文件
    for data_path in data_paths:
        # Find all requested files.
        #找到所有的响应文件，enumerate将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            #如果found不存在
            if not found:
                #将相应的路径赋值到file_path
                file_path = os.path.abspath(os.path.join(data_path, filename))
                #如果存在file_path的话
                if os.path.exists(file_path):
                    #将相应的file_path添加到found_files列表中
                    found_files[index] = file_path

    # Check that all files were found
    #检查是不是找到了全部的文件
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError("Could not find {:}. Searched in data paths: {:}\n{:}".format(filename, data_paths, err_msg))
    #返回相应的found_files列表        
    return found_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
#HostDeviceMem(host_mem, device_mem)
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
#给相应的引擎分配缓冲区，用来保存主机和设备的输入输出数据等
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    #创建一个cuda流
    stream = cuda.Stream()
    for binding in engine:
        #trt.volume用来计算可迭代对象的体积
        #get_binding_shape用来获取相应绑定的维度
        #size表示engine中绑定的所需要的最大维度
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        #get_binding_dtype用来获取相应绑定的数据类型
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        #给主机和设备分配缓冲区
        #cuda.pagelocked_empty给主机分配相关的页面锁定内存
        host_mem = cuda.pagelocked_empty(size, dtype)
        #给设备分配内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        #将分配给设备的内存添加到设备绑定
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        #确定绑定是否是一个输入绑定
        if engine.binding_is_input(binding):
            #如果是的话
            #HostDeviceMem的实现参考common.py
            #将相应的内存地址添加到对应的列表里面
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            #如果不是的话
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
#进行相应的推理过程
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    #将相应的数据从主机内存复制到GPU设备
    #memcpy_htod_async的具体实现参考https://www.osgeo.cn/pycuda/driver.html?highlight=memcpy_htod_async#pycuda.driver.memcpy_htod_async
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    #运行相应的推理过程
    #execute_async的具体实现参考https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html?highlight=execute_async#tensorrt.IExecutionContext.execute_async
    #execute_async对批处理数据执行异步推理
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    #将推理结果传输到主机内存
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    #进行相应的cuda流同步，类似多线程的同步机制
    stream.synchronize()
    # Return only the host outputs.
    #只需要返回主机内存里面的数据即可
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def generate_md5_checksum(local_path):
    """Returns the MD5 checksum of a local file.

    Keyword argument:
    local_path -- path of the file whose checksum shall be generated
    """
    with open(local_path, 'rb') as local_file:
        data = local_file.read()
        import hashlib
        return hashlib.md5(data).hexdigest()


def download_file(local_path, link, checksum_reference=None):
    """Checks if a local file is present and downloads it from the specified path otherwise.
    If checksum_reference is specified, the file's md5 checksum is compared against the
    expected value.

    Keyword arguments:
    local_path -- path of the file whose checksum shall be generated
    link -- link where the file shall be downloaded from if it is not found locally
    checksum_reference -- expected MD5 checksum of the file
    """
    if not os.path.exists(local_path):
        print('Downloading from %s, this may take a while...' % link)
        import wget
        wget.download(link, local_path)
        print()
    if checksum_reference is not None:
        checksum = generate_md5_checksum(local_path)
        if checksum != checksum_reference:
            raise ValueError(
                'The MD5 checksum of local file %s differs from %s, please manually remove \
                 the file and try again.' %
                (local_path, checksum_reference))
    return local_path


# `retry_call` and `retry` are used to wrap the function we want to try multiple times
def retry_call(func, args=[], kwargs={}, n_retries=3):
    """Wrap a function to retry it several times.

    Args:
        func: function to call
        args (List): args parsed to func
        kwargs (Dict): kwargs parsed to func
        n_retries (int): maximum times of tries
    """
    for i_try in range(n_retries):
        try:
            func(*args, **kwargs)
            break
        except:
            if i_try == n_retries - 1:
                raise
            print("retry...")

# Usage: @retry(n_retries)
def retry(n_retries=3):
    """Wrap a function to retry it several times. Decorator version of `retry_call`.

    Args:
        n_retries (int): maximum times of tries

    Usage:
        @retry(n_retries)
        def func(...):
            pass
    """
    def wrapper(func):
        def _wrapper(*args, **kwargs):
            retry_call(func, args, kwargs, n_retries)
        return _wrapper
    return wrapper
