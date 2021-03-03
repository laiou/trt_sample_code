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

# This file contains functions for training a PyTorch MNIST Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import os

from random import randint

# Network
#模型信息类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)
        #模型的前向传播过程
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
#具体的模型类，包括相应的超参数以及训练等过程
class MnistModel(object):
    def __init__(self):
        #相应模型的超参数
        self.batch_size = 64
        self.test_batch_size = 100
        self.learning_rate = 0.0025
        self.sgd_momentum = 0.9
        self.log_interval = 100
        # Fetch MNIST data set.
        #加载相应训练数据
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist/data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600)
        #加载相应的测试数据
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist/data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600)
        self.network = Net()

        self.latest_test_accuracy = 0.0

    # Train the network for one or more epochs, validating after each epoch.
    #模型的训练过程
    def learn(self, num_epochs=2):
        # Train the network for a single epoch
        def train(epoch):
            self.network.train()
            #创建相应的优化器
            optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum)
            #循环遍历batch中的数据
            for batch, (data, target) in enumerate(self.train_loader):
                data, target = Variable(data), Variable(target)
                #进行相应的梯度清0
                optimizer.zero_grad()
                #前向传播过程
                output = self.network(data)
                #计算相应的损失
                loss = F.nll_loss(output, target)
                #进行反向传播
                loss.backward()
                #进行参数更新
                optimizer.step()
                if batch % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data), len(self.train_loader.dataset), 100. * batch / len(self.train_loader), loss.data.item()))

        # Test the network
        #测试相应的模型
        def test(epoch):
            self.network.eval()
            test_loss = 0
            correct = 0
            #循环遍历相应的测试数据
            for data, target in self.test_loader:
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                #进行前向传播
                output = self.network(data)
                #计算损失
                test_loss += F.nll_loss(output, target).data.item()
                #得到最终的预测结果
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()

            test_loss /= len(self.test_loader)
            self.latest_test_accuracy = float(correct) / len(self.test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), 100. * self.latest_test_accuracy))

            

        for e in range(num_epochs):
            train(e + 1)
            test(e + 1)

    # @brief Get the latest accuracy on the test set
    # @pre self.learn.test (and thus self.learn()) need to be run
    def get_latest_test_set_accuracy(self):
        return self.latest_test_accuracy

    def get_weights(self):
        return self.network.state_dict()

    # Retrieve a single sample out of a batch and convert to flattened numpy array
    #将相应的测试图片数据转换成一维数组
    def convert_to_flattened_numpy_array(self, batch_data, batch_target, sample_idx):
        test_case = batch_data.numpy()[sample_idx].ravel().astype(np.float32)
        test_name = batch_target.numpy()[sample_idx]
        return test_case, test_name

    # Generator to loop over every sample in the test set, sample by sample
    #循环遍历每一个测试数据
    def get_all_test_samples(self):
        for data, target in self.test_loader:
            for case_num in range(len(data)):
                #convert_to_flattened_numpy_array参考本文件下的实现
                #将测试图片数据降维，通过生成器迭代遍历每一个数据
                yield self.convert_to_flattened_numpy_array(data, target, case_num)
