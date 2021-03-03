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
import tensorflow as tf
import numpy as np

def process_dataset():
    # Import the data
    #加载数据
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #归一化处理
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    #划分训练和测试的数据量
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    #对加载的数据进行维度上的调整（batch-size,width,height,channel）
    x_train = np.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = np.reshape(x_test, (NUM_TEST, 28, 28, 1))
    #返回相应的训练数据和测试数据以及相应的label
    return x_train, y_train, x_test, y_test
#创建模型的函数
def create_model():
    #Sequential()顺序模型是多个网络层的线性堆叠。参考https://keras.io/zh/getting-started/sequential-model-guide/
    model = tf.keras.models.Sequential()
    #通过add方法将各个层次添加到Sequential()中完成整个模型的构建
    model.add(tf.keras.layers.InputLayer(input_shape=[28,28, 1]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
#保存模型的函数 
def save(model, filename):
    # First freeze the graph and remove training nodes.
    #获取模型输出节点的名称
    output_names = model.output.op.name
    #创建一个session会话
    sess = tf.keras.backend.get_session()
    #convert_variables_to_constants会将计算图中的变量取值以常量的形式保存，参考会将计算图中的变量取值以常量的形式保存
    #这里只保存了计算图的部分节点，有些辅助结点没有保存，同时通过指定保存的节点名称来设置
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
    #移除训练相关的结点，参考remove_training_nodes
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    # Save the model
    #保存模型到相应的文件
    with open(filename, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())

def main():
    #加载数据
    x_train, y_train, x_test, y_test = process_dataset()
    #创建模型
    model = create_model()
    # Train the model on the data
    #将相应数据传输到模型
    model.fit(x_train, y_train, epochs = 5, verbose = 1)
    # Evaluate the model on test data
    #验证训练效果
    model.evaluate(x_test, y_test)
    #保存相应的模型
    save(model, filename="models/lenet5.pb")

if __name__ == '__main__':
    main()
