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

import sys
import os
import ctypes
import time
import argparse
import glob

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import numpy as np
import tensorrt as trt
from PIL import Image

# Utility functions
#具体的参考utils文件夹下的相关实现
import utils.inference as inference_utils # TRT/TF inference wrappers
import utils.model as model_utils # UFF conversion
import utils.mAP as voc_mAP_utils # mAP computation
import utils.voc as voc_utils # VOC dataset descriptors
import utils.coco as coco_utils # COCO dataset descriptors
from utils.paths import PATHS # Path management


# VOC and COCO label lists
#VOC和COCO数据集的label列表
VOC_CLASSES = voc_utils.VOC_CLASSES_LIST
COCO_LABELS = coco_utils.COCO_CLASSES_LIST

# Model used for inference
#用于推理的模型名称
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'

# Precision command line argument -> TRT Engine datatype
#相应的trt引擎的数据类型
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

# Layout of TensorRT network output metadata
#tensorrt的网络输出数据格式
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

#检测类
class Detection(object):
    """Describes detection for VOC detection file.

    During evaluation of model on VOC, we save objects detected to result
    files, with one file per each class. One line in such file corresponds
    to one object that is detected in an image. The Detection class describes
    one such detection.

    Attributes:
        image_number (str): number of image from VOC dataset
        confidence (float): confidence score for detection
        xmin (float): bounding box min x coordinate
        ymin (float): bounding box min y coordinate
        xmax (float): bounding box max x coordinate
        ymax (float): bounding box max y coordinate
    """
    #声明相应的参数
    def __init__(self, image_number, confidence, xmin, ymin, xmax, ymax):
        #数据集图片数量
        self.image_number = image_number
        #相应的预测置信度
        self.confidence = confidence
        #框坐标
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __repr__(self):
        return "{} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
            self.image_number, self.confidence,
            self.xmin, self.ymin, self.xmax, self.ymax
        )

    def write_to_file(self, f):
        """Adds detection corresponding to Detection object to file f.

        Args:
            f (file): detection results file
        """
        f.write(self.__repr__())


def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
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

def analyze_tensorrt_prediction(detection_out, pred_start_idx):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)

    xmin = float(xmin) * model_utils.ModelData.get_input_width()
    ymin = float(ymin) * model_utils.ModelData.get_input_height()
    xmax = float(xmax) * model_utils.ModelData.get_input_width()
    ymax = float(ymax) * model_utils.ModelData.get_input_height()

    return image_id, label, confidence, xmin, ymin, xmax, ymax
#produce_tensorrt_detections(detection_files,trt_inference_wrapper, parsed['max_batch_size'],voc_image_numbers, voc_image_path)
#处理相应的tensorrt的推理输出
def produce_tensorrt_detections(detection_files, trt_inference_wrapper, max_batch_size,
    image_numbers, image_path):
    """Fetches output from TensorRT model, and saves it to results file.

    The output of TensorRT model is a pair of:
      * location byte array that contains detection metadata,
        which is layout according to TRT_PREDICTION_LAYOUT
      * number of detections returned by NMS

    TRT_PREDICTION_LAYOUT fields correspond to Tensorflow ones as follows:
      label -> detection_classes
      confidence -> detection_scores
      xmin, ymin, xmax, ymax -> detection_boxes

    The number of detections correspond to num_detection Tensorflow output.

    Tensorflow output semantics is more throughly explained in
    produce_tensorflow_detections().

    This function iterates over all VOC images, feeding each one
    into TensotRT model, fetching object detections
    from each output, converting them to Detection object,
    and saving to detection result file.

    Args:
        detection_files (dict): dictionary that maps class labels to
            class result files
        trt_inference_wrapper (inference_utils.TRTInference):
            internal Python class wrapping TensorRT inferece
            setup/run code
        batch_size (int): batch size used for inference
        image_numbers [str]: VOC image numbers to use for inference
        image_path (str): Python string, which stores path to VOC image file,
            when you do image_path.format(voc_mage_number)
    """
    #获取相应的图片数量
    total_imgs = len(image_numbers)
    #随机取每个batch-size数据
    for idx in range(0, len(image_numbers), max_batch_size):
        #抽取一个batch_size的数据
        imgs = image_numbers[idx:idx+max_batch_size]
        batch_size = len(imgs)
        print("Infering image {}/{}".format(idx+1, total_imgs))
        #提取相应的具体路径
        image_paths = [image_path.format(img) for img in imgs]
        #infer_batch具体的推理，参考utils/inference.py的实现
        detections, keep_count = trt_inference_wrapper.infer_batch(image_paths)
        #获取预测输出的数据长度
        prediction_fields = len(TRT_PREDICTION_LAYOUT)
        #遍历每一张batch中的图片数据
        for img_idx, img_number in enumerate(imgs):
            #提取相对应的预测值索引
            img_predictions_start_idx = prediction_fields * keep_count[img_idx] * img_idx
            #抽取相应的具体输出值
            for det in range(int(keep_count[img_idx])):
                #抽取相应的具体输出，analyze_tensorrt_prediction参考本文件下的实现
                _, label, confidence, xmin, ymin, xmax, ymax = \
                    analyze_tensorrt_prediction(detections, img_predictions_start_idx + det * prediction_fields)
                if confidence > 0.0:
                    #提取相应的label列表
                    label_name = voc_utils.coco_label_to_voc_label(COCO_LABELS[label])
                    #如果检测到相应的label，将结果写到文件中保存
                    if label_name:
                        det_file = detection_files[label_name]
                        detection = Detection(
                            img_number,
                            confidence,
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                        )
                        detection.write_to_file(det_file)
#解析tensorflow的预测输出
def produce_tensorflow_detections(detection_files, tf_inference_wrapper, batch_size,
    image_numbers, image_path):
    """Fetches output from Tensorflow model, and saves it to results file.

    The format of output from Tensorflow is output_dict Python
    dictionary containing following fields:
        num_detections: maximum number of detections keeped per image
        detection_classes: label of classes detected
        detection_scores: confidences for detections
        detection_boxes: bounding box coordinates for detections,
            in format (ymin, xmin, ymax, xmax)

    This function iterates over all VOC images, feeding each one
    into Tensorflow model, fetching object detections
    from each output, converting them to Detection object,
    and saving to detection result file.

    Args:
        detection_files (dict): dictionary that maps class labels to
            class result files
        tf_inference_wrapper (inference_utils.TensorflowInference):
            internal Python class wrapping Tensorflow inferece
            setup/run code
        batch_size (int): batch size used for inference
        image_numbers [str]: VOC image numbers to use for inference
        image_path (str): Python string, which stores path to VOC image file,
            when you do image_path.format(voc_mage_number)
    """
    total_imgs = len(image_numbers)
    for idx in range(0, len(image_numbers), batch_size):
        print("Infering image {}/{}".format(idx+1, total_imgs))

        imgs = image_numbers[idx:idx+batch_size]
        image_paths = [image_path.format(img) for img in imgs]
        output_dict = tf_inference_wrapper.infer_batch(image_paths)

        keep_count = output_dict['num_detections']
        for img_idx, img_number in enumerate(imgs):
            for det in range(int(keep_count[img_idx])):
                label = output_dict['detection_classes'][img_idx][det]
                confidence = output_dict['detection_scores'][img_idx][det]
                bbox = output_dict['detection_boxes'][img_idx][det]

                # Output bounding boxes are in [0, 1] format,
                # here we rescale them to pixel [0, 255] format
                ymin, xmin, ymax, xmax = bbox
                xmin = float(xmin) * model_utils.ModelData.get_input_width()
                ymin = float(ymin) * model_utils.ModelData.get_input_height()
                xmax = float(xmax) * model_utils.ModelData.get_input_width()
                ymax = float(ymax) * model_utils.ModelData.get_input_height()

                # Detection is saved only if confidence is bigger than zero
                if confidence > 0.0:
                    # Model was trained on COCO, so we need to convert label to VOC one
                    label_name = voc_utils.coco_label_to_voc_label(COCO_LABELS[label])
                    if label_name: # Checks for label_name correctness
                        det_file = detection_files[label_name]
                        detection = Detection(
                            img_number,
                            confidence,
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                        )
                        detection.write_to_file(det_file)
#判断是不是跳过推理
def should_skip_inference(parsed_args):
    """Checks if inference should be skipped.

    When evaluating on VOC, if results from some earlier run
    of the script exist, we can reuse them to evaluate VOC mAP.
    The user can overwrite this behavior by supplying -f flag to the script.

    Args:
        parsed_args (dict): commandline arguments parsed by
            parse_commandline_arguments()

    Returns:
        bool: if True, script skips inference
    """
    skip_inference = True
    for voc_class in VOC_CLASSES:
        voc_class_detection_file = \
            os.path.join(parsed_args['results_dir'], 'det_test_{}.txt'.format(voc_class))
        if os.path.exists(voc_class_detection_file) and not parsed_args['force_inference']:
            continue
        else:
            skip_inference = False
    if skip_inference:
        print("Model detections present - skipping inference. To avoid this, use -f flag.")
    return skip_inference
#对图片数据进行预处理操作
def preprocess_voc():
    """Resizes all VOC images to 300x300 and saves them into .ppm files.

    This script assumes all images fetched to network in batches have size 300x300,
    so in this function we preproceess all VOC images to fit that format.
    """
    #获取图片数据的根目录
    voc_root = PATHS.get_voc_dir_path()
    #glob.glob用它可以查找符合特定规则的文件路径名。同时获取所有的匹配路径
    voc_jpegs = glob.glob(
        os.path.join(voc_root, 'JPEGImages', '*.jpg'))
    voc_ppms = glob.glob(
        os.path.join(voc_root, 'PPMImages', '*.ppm'))

    # Check if preprocessing is needed by comparing
    # image names between JPEGImages and PPMImages
    voc_jpegs_basenames = \
        [os.path.splitext(os.path.basename(p))[0] for p in voc_jpegs]
    voc_ppms_basenames = \
        [os.path.splitext(os.path.basename(p))[0] for p in voc_ppms]
    # If lists are not the same, preprocessing is needed
    if sorted(voc_jpegs_basenames) != sorted(voc_ppms_basenames):
        print("Preprocessing VOC dataset. It may take few minutes.")
        # Make PPMImages directory if it doesn't exist
        voc_ppms_path = PATHS.get_voc_ppm_img_path()
        if not os.path.exists(os.path.dirname(voc_ppms_path)):
            os.makedirs(os.path.dirname(voc_ppms_path))
        # For each .jpg file, make a resized
        # .ppm copy to fit model input expectations
        #对每张jpg图片进行处理
        for voc_jpeg_path in voc_jpegs:
            voc_jpeg_basename = os.path.basename(voc_jpeg_path)
            voc_ppm_path = voc_ppms_path.format(
                os.path.splitext(voc_jpeg_basename)[0])
            if not os.path.exists(voc_ppm_path):
                img_pil = Image.open(voc_jpeg_path)
                #先进行resize调整大小
                img_pil = img_pil.resize(
                    size=(
                        model_utils.ModelData.get_input_width(),
                        model_utils.ModelData.get_input_height()),
                    resample=Image.BILINEAR
                )
                #然后保存成ppm文件
                img_pil.save(voc_ppm_path)

def adjust_paths(args):
    """Adjust all file/directory paths, arguments passed by user.

    During script launch, user can pass several arguments to the script
    (e.g. --workspace_dir, --voc_dir), that define where script will look
    for the files needed for execution. This function adjusts internal
    Paths Python datastructure to accomodate for changes from defaults
    requested by user through appropriate command line arguments.

    Args:
        args (argparse.Namespace): parsed user arguments
    """
    if args.voc_dir:
        PATHS.set_voc_dir_path(args.voc_dir)
    if args.workspace_dir:
        PATHS.set_workspace_dir_path(args.workspace_dir)
    if not os.path.exists(PATHS.get_workspace_dir_path()):
        os.makedirs(PATHS.get_workspace_dir_path())
#解析相应的命令行参数
def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    #声明一个参数解析器
    parser = argparse.ArgumentParser(description='Run object detection evaluation on VOC2007 dataset.')
    parser.add_argument('inference_backend', metavar='INFERENCE_BACKEND',
        type=str, choices=['tensorrt', 'tensorflow'], default='tensorrt', nargs='?',
        help='inference backend to run evaluation with')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16], default=32,
        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=64,
        help='max TensorRT engine batch size')
    parser.add_argument('-f', '--force_inference', action='store_true',
        help='force model inference even if detections exist')
    parser.add_argument('-w', '--workspace_dir',
        help='sample workspace directory')
    parser.add_argument('-voc', '--voc_dir',
        help='VOC2007 root directory')

    # Parse arguments passed
    #解析传递的参数
    args = parser.parse_args()

    # Adjust global Paths data structure
    #调整全局路径结构，具体参考本文件下的实现
    adjust_paths(args)

    # Verify Paths after adjustments. This also exits script if verification fails
    PATHS.verify_all_paths(should_verify_voc=True)

    # Fetch directory to save inference results to, create it if it doesn't exist
    trt_engine_datatype = None
    trt_engine_path = None
    if args.inference_backend == 'tensorrt':
        # In case of TensorRT we also fetch engine data type and engine path
        trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
        trt_engine_path = PATHS.get_engine_path(trt_engine_datatype,
            args.max_batch_size)
        if not os.path.exists(os.path.dirname(trt_engine_path)):
            os.makedirs(os.path.dirname(trt_engine_path))
        results_dir = PATHS.get_voc_model_detections_path('tensorrt',
            trt_engine_datatype)
    elif args.inference_backend == 'tensorflow':
        results_dir = PATHS.get_voc_model_detections_path('tensorflow')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Return parsed arguments for further functions to use
    parsed = {
        'inference_backend': args.inference_backend,
        'max_batch_size': args.max_batch_size,
        'force_inference': args.force_inference,
        'results_dir': results_dir,
        'trt_engine_path': trt_engine_path,
        'trt_engine_datatype': trt_engine_datatype
    }
    return parsed


if __name__ == '__main__':
    # Parse command line arguments
    #解析命令行参数，参考本文件下的实现
    parsed = parse_commandline_arguments()

    # Check if inference should be skipped (if model inference
    # results are already computed, we don't need to recompute
    # them for VOC mAP computation)
    #判断是否应该跳过推理
    #should_skip_inference参考本文件下的实现
    skip_inference = should_skip_inference(parsed)
    # And if inference will not be skipped, then we
    # create files to store its results in
    #创建相应的文件存储推理结果
    detection_files = {}
    if not skip_inference:
        for voc_class in VOC_CLASSES:
            detection_files[voc_class] = open(
                os.path.join(
                    parsed['results_dir'], 'det_test_{}.txt'.format(voc_class)
                ), 'w'
            )

    # Fetch frozen model .pb path...
    #获取pb模型的路径
    ssd_model_pb_path = PATHS.get_model_pb_path(MODEL_NAME)
    # ...and .uff path, if needed (converting .pb to .uff if not already done)
    if parsed['inference_backend'] == 'tensorrt':
        #获取uff模型的路径
        ssd_model_uff_path = PATHS.get_model_uff_path(MODEL_NAME)
        #没有uff模型的话，将pb模型转换成uff模型
        if not os.path.exists(ssd_model_uff_path):
            model_utils.prepare_ssd_model(MODEL_NAME)

    # This block of code sets up and performs inference, if needed
    if not skip_inference:
        # Preprocess VOC dataset if necessary by resizing images
        #对图片数据进行预处理，具体参考本文件下的实现
        preprocess_voc()

        # Fetch image list and input .ppm files path
        #获取图片数据列表和相应的ppm路径
        with open(PATHS.get_voc_image_set_path(), 'r') as f:
            voc_image_numbers = f.readlines()
            voc_image_numbers = [line.strip() for line in voc_image_numbers]
        voc_image_path = PATHS.get_voc_ppm_img_path()

        # Tensorflow and TensorRT paths are a little bit different,
        # so we must treat each one individually
        #判断是使用tensorrt还是 tensorflow来进行推理
        if parsed['inference_backend'] == 'tensorrt':
            # TRTInference initialization initializes
            # all TensorRT structures, creates engine if it doesn't
            # already exist and finally saves it to file for future uses
            #TRTInference参考utils/inference.py下的实现
            trt_inference_wrapper = inference_utils.TRTInference(
                parsed['trt_engine_path'], ssd_model_uff_path,
                parsed['trt_engine_datatype'], parsed['max_batch_size'])
            # Outputs from TensorRT are handled differently than
            # outputs from Tensorflow, that's why we use another
            # function to produce the detections from them
            #produce_tensorrt_detections参考本文件下的实现
            #解析tensorrt推理的输出
            #参考本文件下的实现
            produce_tensorrt_detections(detection_files,
                trt_inference_wrapper, parsed['max_batch_size'],
                voc_image_numbers, voc_image_path)
            #如果是利用tensorflow来推理的话
        elif parsed['inference_backend'] == 'tensorflow':
            # In case of Tensorflow all we need to
            # initialize inference is frozen model...
            #TensorflowInference参考inference.py
            tf_inference_wrapper = \
                inference_utils.TensorflowInference(ssd_model_pb_path)
            # ...and after initializing it, we can
            # proceed to producing detections
            #解析tensorflow的预测输出，参考本文件下的实现
            produce_tensorflow_detections(detection_files,
                tf_inference_wrapper, parsed['max_batch_size'],
                voc_image_numbers, voc_image_path)


    # Flush detection to files to make sure evaluation is correct
    #刷新对应的存储文件，保证检测的结果
    #flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入
    #因为这个时候相应的文件还没有关闭，不用flush的话，可能还有内容在缓冲区，没有写进去
    for key in detection_files:
        detection_files[key].flush()

    # Do mAP computation based on saved detections
    #计算相应的map，参考utils/mAP.py
    voc_mAP_utils.do_python_eval(parsed['results_dir'])

    # Close detection files, they are not needed anymore
    #关闭文件
    for key in detection_files:
        detection_files[key].close()
