#!/usr/bin/env python3
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

import os
import ctypes
import time
import sys
import argparse

import numpy as np
from PIL import Image
import tensorrt as trt
#具体参考utils目录下的相关实现
import utils.inference as inference_utils # TRT/TF inference wrappers
import utils.model as model_utils # UFF conversion
import utils.boxes as boxes_utils # Drawing bounding boxes
import utils.coco as coco_utils # COCO dataset descriptors
from utils.paths import PATHS # Path management


# COCO label list
#COCO数据集的label列表
COCO_LABELS = coco_utils.COCO_CLASSES_LIST

# Model used for inference
#模型文件的名字
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'

# Confidence threshold for drawing bounding box
#画检测框的置信度阈值
VISUALIZATION_THRESHOLD = 0.5

# Precision command line argument -> TRT Engine datatype
#tensorrt的引擎数据类型
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

# Layout of TensorRT network output metadata
#tensorrt输出的数据格式
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

#返回相应预测中的值，根据索引抽取对应位置上的预测输出
def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    The prediction fields layout is described in TRT_PREDICTION_LAYOUT.

    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.

    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out

    Returns:
        Prediction field corresponding to given data.
    """
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]
#后处理的过程
def analyze_prediction(detection_out, pred_start_idx, img_pil):
    #抽取预测中的相关值
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    #判断置信度阈值
    if confidence > VISUALIZATION_THRESHOLD:
        #得到预测的类别
        class_name = COCO_LABELS[label]
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
        #在图片上画出预测框
        #draw_bounding_boxes_on_image参考boxes.py
        boxes_utils.draw_bounding_boxes_on_image(
            img_pil, np.array([[ymin, xmin, ymax, xmax]]),
            display_str_list=["{}: {}".format(
                class_name, confidence_percentage)],
            color=coco_utils.COCO_COLORS[label]
        )

#解析命令行参数
def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(description='Run object detection inference on input image.')
    parser.add_argument('input_img_path', metavar='INPUT_IMG_PATH',
        help='an image file to run inference on')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16], default=32,
        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
        help='max TensorRT engine batch size')
    parser.add_argument('-w', '--workspace_dir',
        help='sample workspace directory')
    parser.add_argument("-o", "--output",
        help="path of the output file",
        default=os.path.join(PATHS.get_sample_root(), "image_inferred.jpg"))

    # Parse arguments passed
    args = parser.parse_args()

    # Set workspace dir path if passed by user
    if args.workspace_dir:
        PATHS.set_workspace_dir_path(args.workspace_dir)

    try:
        os.makedirs(PATHS.get_workspace_dir_path())
    except:
        pass

    # Verify Paths after adjustments. This also exits script if verification fails
    PATHS.verify_all_paths()

    # Fetch TensorRT engine path and datatype
    #获取tensorrt引擎的数据类型和相应的引擎文件的路径
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    #get_engine_path参考utils/paths.py下的实现，获取引擎文件的路径
    args.trt_engine_path = PATHS.get_engine_path(args.trt_engine_datatype,
        args.max_batch_size)
    try:
        os.makedirs(os.path.dirname(args.trt_engine_path))
    except:
        pass

    return args

def main():
    # Parse command line arguments
    #解析命令行参数，具体参考本文件下的实现
    args = parse_commandline_arguments()

    # Fetch .uff model path, convert from .pb
    # if needed, using prepare_ssd_model
    #获取相应的uff模型的路径
    #get_model_uff_path参考utils/paths.py下的实现
    ssd_model_uff_path = PATHS.get_model_uff_path(MODEL_NAME)
    #如果不存在uff模型
    if not os.path.exists(ssd_model_uff_path):
        #从pb模型完成到uff模型的转换
        #prepare_ssd_model参考utils/model.py下的实现
        model_utils.prepare_ssd_model(MODEL_NAME)

    # Set up all TensorRT data structures needed for inference
    #建立推理所需要的数据结构
    #具体参考utils/inference.py下的实现
    trt_inference_wrapper = inference_utils.TRTInference(
        args.trt_engine_path, ssd_model_uff_path,
        trt_engine_datatype=args.trt_engine_datatype,
        batch_size=args.max_batch_size)

    # Start measuring time
    inference_start_time = time.time()

    # Get TensorRT SSD model output
    #获取相应的推理输出
    #参考utils/inference.py
    detection_out, keep_count_out = \
        trt_inference_wrapper.infer(args.input_img_path)

    # Make PIL.Image for drawing bounding boxes and
    # let analyze_prediction() draw them based on model output
    img_pil = Image.open(args.input_img_path)
    prediction_fields = len(TRT_PREDICTION_LAYOUT)
    for det in range(int(keep_count_out[0])):
        #analyze_prediction参考本文件的实现
        analyze_prediction(detection_out, det * prediction_fields, img_pil)

    # Output total [img load + inference + drawing bboxes] time
    print("Total time taken for one image: {} ms\n".format(
        int(round((time.time() - inference_start_time) * 1000))))

    # Save output image and output path
    #保存相应的结果
    img_pil.save(args.output)
    print("Saved output image to: {}".format(args.output))


if __name__ == '__main__':
    main()
