# Copyright 2019 Xilinx Inc.
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

# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from pytorch_nndct.nn import QuantStub, DeQuantStub
from pytorch_nndct.nn.modules import functional

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.module_0 = torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=[6, 6], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[0]/Conv2d[conv]/input.3
        self.module_1 = torch.nn.BatchNorm2d(num_features=48, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[0]/torch.nn.BatchNorm2d2d[bn]/input.5
        self.module_2 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[0]/LeakyReLU[act]/input.7
        self.module_3 = torch.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[1]/Conv2d[conv]/input.9
        self.module_4 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[1]/torch.nn.BatchNorm2d2d[bn]/input.11
        self.module_5 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[1]/LeakyReLU[act]/input.13
        self.module_6 = torch.nn.Conv2d(in_channels=96, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[2]/Conv[cv1]/Conv2d[conv]/input.15
        self.module_7 = torch.nn.BatchNorm2d(num_features=48, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[2]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.17
        self.module_8 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[2]/Conv[cv1]/LeakyReLU[act]/input.19
        self.module_21 = torch.nn.Conv2d(in_channels=96, out_channels=48, kernel_size=[1, 1], stride=[1, 1],
                                         padding=[0, 0], dilation=[1, 1], groups=1,
                                         bias=False)  # Model::Model/C3[model]/C3[2]/Conv[cv2]/Conv2d[conv]/input.43
        self.module_22 = torch.nn.BatchNorm2d(num_features=48, eps=0.001,
                                              momentum=0.03)  # Model::Model/C3[model]/C3[2]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.45
        self.module_23 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                            inplace=True)  # Model::Model/C3[model]/C3[2]/Conv[cv2]/LeakyReLU[act]/13316
        self.module_24 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1],
                                         padding=[0, 0], dilation=[1, 1], groups=1,
                                         bias=False)  # Model::Model/C3[model]/C3[2]/Conv[cv3]/Conv2d[conv]/input.49
        self.module_25 = torch.nn.BatchNorm2d(num_features=96, eps=0.001,
                                              momentum=0.03)  # Model::Model/C3[model]/C3[2]/Conv[cv3]/torch.nn.BatchNorm2d2d[bn]/input.51
        self.module_26 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                            inplace=True)  # Model::Model/C3[model]/C3[2]/Conv[cv3]/LeakyReLU[act]/input.53

        self.module_9 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.21
        self.module_10 = torch.nn.BatchNorm2d(num_features=48, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.23
        self.module_11 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.25
        self.module_12 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.27
        self.module_13 = torch.nn.BatchNorm2d(num_features=48, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.29
        self.module_14 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/13231
        self.module_15 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.33
        self.module_16 = torch.nn.BatchNorm2d(num_features=48, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.35
        self.module_17 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.37
        self.module_18 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.39
        self.module_19 = torch.nn.BatchNorm2d(num_features=48, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.41
        self.module_20 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/13287
        self.module_27 = torch.nn.Conv2d(in_channels=96, out_channels=192, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[3]/Conv2d[conv]/input.55
        self.module_28 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[3]/torch.nn.BatchNorm2d2d[bn]/input.57
        self.module_29 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[3]/LeakyReLU[act]/input.59
        self.module_30 = torch.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Conv[cv1]/Conv2d[conv]/input.61
        self.module_31 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.63
        self.module_32 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Conv[cv1]/LeakyReLU[act]/input.65
        self.module_57 = torch.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[1, 1], stride=[1, 1],
                                         padding=[0, 0], dilation=[1, 1], groups=1,
                                         bias=False)  # Model::Model/C3[model]/C3[4]/Conv[cv2]/Conv2d[conv]/input.113
        self.module_58 = torch.nn.BatchNorm2d(num_features=96, eps=0.001,
                                              momentum=0.03)  # Model::Model/C3[model]/C3[4]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.115
        self.module_59 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                            inplace=True)  # Model::Model/C3[model]/C3[4]/Conv[cv2]/LeakyReLU[act]/13651
        self.module_60 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1],
                                         padding=[0, 0], dilation=[1, 1], groups=1,
                                         bias=False)  # Model::Model/C3[model]/C3[4]/Conv[cv3]/Conv2d[conv]/input.119
        self.module_61 = torch.nn.BatchNorm2d(num_features=192, eps=0.001,
                                              momentum=0.03)  # Model::Model/C3[model]/C3[4]/Conv[cv3]/torch.nn.BatchNorm2d2d[bn]/input.121
        self.module_62 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                            inplace=True)  # Model::Model/C3[model]/C3[4]/Conv[cv3]/LeakyReLU[act]/input.123

        self.module_33 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.67
        self.module_34 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.69
        self.module_35 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.71
        self.module_36 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.73
        self.module_37 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.75
        self.module_38 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/13454
        self.module_39 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.79
        self.module_40 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.81
        self.module_41 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.83
        self.module_42 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.85
        self.module_43 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.87
        self.module_44 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/13510
        self.module_45 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.91
        self.module_46 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.93
        self.module_47 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.95
        self.module_48 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.97
        self.module_49 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.99
        self.module_50 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/13566
        self.module_51 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.103
        self.module_52 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.105
        self.module_53 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.107
        self.module_54 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.109
        self.module_55 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.111
        self.module_56 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/13622
        self.module_63 = torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[5]/Conv2d[conv]/input.125
        self.module_64 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[5]/torch.nn.BatchNorm2d2d[bn]/input.127
        self.module_65 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[5]/LeakyReLU[act]/input.129
        self.module_66 = torch.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Conv[cv1]/Conv2d[conv]/input.131
        self.module_67 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.133
        self.module_68 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Conv[cv1]/LeakyReLU[act]/input.135
        self.module_105 = torch.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[6]/Conv[cv2]/Conv2d[conv]/input.207
        self.module_106 = torch.nn.BatchNorm2d(num_features=192, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[6]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.209
        self.module_107 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[6]/Conv[cv2]/LeakyReLU[act]/14098
        self.module_108 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[6]/Conv[cv3]/Conv2d[conv]/input.213
        self.module_109 = torch.nn.BatchNorm2d(num_features=384, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[6]/Conv[cv3]/torch.nn.BatchNorm2d2d[bn]/input.215
        self.module_110 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[6]/Conv[cv3]/LeakyReLU[act]/input.217

        self.module_69 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.137
        self.module_70 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.139
        self.module_71 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.141
        self.module_72 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.143
        self.module_73 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.145
        self.module_74 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/13789
        self.module_75 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.149
        self.module_76 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.151
        self.module_77 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.153
        self.module_78 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.155
        self.module_79 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.157
        self.module_80 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/13845
        self.module_81 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.161
        self.module_82 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.163
        self.module_83 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.165
        self.module_84 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.167
        self.module_85 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.169
        self.module_86 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/13901
        self.module_87 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.173
        self.module_88 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.175
        self.module_89 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.177
        self.module_90 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.179
        self.module_91 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.181
        self.module_92 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/13957
        self.module_93 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv1]/Conv2d[conv]/input.185
        self.module_94 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.187
        self.module_95 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv1]/LeakyReLU[act]/input.189
        self.module_96 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv2]/Conv2d[conv]/input.191
        self.module_97 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.193
        self.module_98 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv2]/LeakyReLU[act]/14013
        self.module_99 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv1]/Conv2d[conv]/input.197
        self.module_100 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.199
        self.module_101 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv1]/LeakyReLU[act]/input.201
        self.module_102 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv2]/Conv2d[conv]/input.203
        self.module_103 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.205
        self.module_104 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv2]/LeakyReLU[act]/14069
        self.module_111 = torch.nn.Conv2d(in_channels=384, out_channels=768, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[7]/Conv2d[conv]/input.219
        self.module_112 = torch.nn.BatchNorm2d(num_features=768, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[7]/torch.nn.BatchNorm2d2d[bn]/input.221
        self.module_113 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[7]/LeakyReLU[act]/input.223
        self.module_114 = torch.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[8]/Conv[cv1]/Conv2d[conv]/input.225
        self.module_115 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[8]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.227
        self.module_116 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[8]/Conv[cv1]/LeakyReLU[act]/input.229
        self.module_129 = torch.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[8]/Conv[cv2]/Conv2d[conv]/input.253
        self.module_130 = torch.nn.BatchNorm2d(num_features=384, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[8]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.255
        self.module_131 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[8]/Conv[cv2]/LeakyReLU[act]/14321
        self.module_132 = torch.nn.Conv2d(in_channels=768, out_channels=768, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[8]/Conv[cv3]/Conv2d[conv]/input.259
        self.module_133 = torch.nn.BatchNorm2d(num_features=768, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[8]/Conv[cv3]/torch.nn.BatchNorm2d2d[bn]/input.261
        self.module_134 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[8]/Conv[cv3]/LeakyReLU[act]/input.263

        self.module_117 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.231
        self.module_118 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.233
        self.module_119 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.235
        self.module_120 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.237
        self.module_121 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.239
        self.module_122 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/14236
        self.module_123 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.243
        self.module_124 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.245
        self.module_125 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.247
        self.module_126 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.249
        self.module_127 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.251
        self.module_128 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/14292
        self.module_135 = torch.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/SPPF[model]/SPPF[9]/Conv[cv1]/Conv2d[conv]/input.265
        self.module_136 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/SPPF[model]/SPPF[9]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.267
        self.module_137 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/SPPF[model]/SPPF[9]/Conv[cv1]/LeakyReLU[act]/14378
        self.module_138 = torch.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #Model::Model/SPPF[model]/SPPF[9]/MaxPool2d[m1]/14392
        self.module_139 = torch.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #Model::Model/SPPF[model]/SPPF[9]/MaxPool2d[m2]/14406
        self.module_140 = torch.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #Model::Model/SPPF[model]/SPPF[9]/MaxPool2d[m3]/14420
        self.module_141 = torch.nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/SPPF[model]/SPPF[9]/Conv[cv2]/Conv2d[conv]/input.271
        self.module_142 = torch.nn.BatchNorm2d(num_features=768, eps=0.001, momentum=0.03) #Model::Model/SPPF[model]/SPPF[9]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.273
        self.module_143 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/SPPF[model]/SPPF[9]/Conv[cv2]/LeakyReLU[act]/input.275
        self.module_144 = torch.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[10]/Conv2d[conv]/input.277
        self.module_145 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[10]/torch.nn.BatchNorm2d2d[bn]/input.279
        self.module_146 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[10]/LeakyReLU[act]/input.281
        self.module_147 = torch.nn.Conv2d(in_channels=768, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[13]/Conv[cv1]/Conv2d[conv]/input.285
        self.module_148 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[13]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.287
        self.module_149 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[13]/Conv[cv1]/LeakyReLU[act]/input.289
        self.module_162 = torch.nn.Conv2d(in_channels=768, out_channels=192, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[13]/Conv[cv2]/Conv2d[conv]/input.313
        self.module_163 = torch.nn.BatchNorm2d(num_features=192, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[13]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.315
        self.module_164 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[13]/Conv[cv2]/LeakyReLU[act]/14647
        self.module_165 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[13]/Conv[cv3]/Conv2d[conv]/input.319
        self.module_166 = torch.nn.BatchNorm2d(num_features=384, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[13]/Conv[cv3]/torch.nn.BatchNorm2d2d[bn]/input.321
        self.module_167 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[13]/Conv[cv3]/LeakyReLU[act]/input.323

        self.module_150 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.291
        self.module_151 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.293
        self.module_152 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.295
        self.module_153 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.297
        self.module_154 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.299
        self.module_155 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.301
        self.module_156 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.303
        self.module_157 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.305
        self.module_158 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.307
        self.module_159 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.309
        self.module_160 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.311
        self.module_161 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/14620
        self.module_168 = torch.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[14]/Conv2d[conv]/input.325
        self.module_169 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[14]/torch.nn.BatchNorm2d2d[bn]/input.327
        self.module_170 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[14]/LeakyReLU[act]/input.329
        self.module_171 = torch.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[17]/Conv[cv1]/Conv2d[conv]/input.333
        self.module_172 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[17]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.335
        self.module_173 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[17]/Conv[cv1]/LeakyReLU[act]/input.337
        self.module_186 = torch.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[17]/Conv[cv2]/Conv2d[conv]/input.361
        self.module_187 = torch.nn.BatchNorm2d(num_features=96, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[17]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.363
        self.module_188 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[17]/Conv[cv2]/LeakyReLU[act]/14874
        self.module_189 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[17]/Conv[cv3]/Conv2d[conv]/input.367
        self.module_190 = torch.nn.BatchNorm2d(num_features=192, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[17]/Conv[cv3]/torch.nn.BatchNorm2d2d[bn]/input.369
        self.module_191 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[17]/Conv[cv3]/LeakyReLU[act]/input.371

        self.module_174 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.339
        self.module_175 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.341
        self.module_176 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.343
        self.module_177 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.345
        self.module_178 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.347
        self.module_179 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.349
        self.module_180 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.351
        self.module_181 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.353
        self.module_182 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.355
        self.module_183 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.357
        self.module_184 = torch.nn.BatchNorm2d(num_features=96, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.359
        self.module_185 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/14847
        self.module_192 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[18]/Conv2d[conv]/input.373
        self.module_193 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[18]/torch.nn.BatchNorm2d2d[bn]/input.375
        self.module_194 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[18]/LeakyReLU[act]/14931
        self.module_195 = torch.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[20]/Conv[cv1]/Conv2d[conv]/input.379
        self.module_196 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[20]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.381
        self.module_197 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[20]/Conv[cv1]/LeakyReLU[act]/input.383
        self.module_210 = torch.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[20]/Conv[cv2]/Conv2d[conv]/input.407
        self.module_211 = torch.nn.BatchNorm2d(num_features=192, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[20]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.409
        self.module_212 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[20]/Conv[cv2]/LeakyReLU[act]/15096
        self.module_213 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[20]/Conv[cv3]/Conv2d[conv]/input.413
        self.module_214 = torch.nn.BatchNorm2d(num_features=384, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[20]/Conv[cv3]/torch.nn.BatchNorm2d2d[bn]/input.415
        self.module_215 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[20]/Conv[cv3]/LeakyReLU[act]/input.417

        self.module_198 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.385
        self.module_199 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.387
        self.module_200 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.389
        self.module_201 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.391
        self.module_202 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.393
        self.module_203 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.395
        self.module_204 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.397
        self.module_205 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.399
        self.module_206 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.401
        self.module_207 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.403
        self.module_208 = torch.nn.BatchNorm2d(num_features=192, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.405
        self.module_209 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/15069
        self.module_216 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/Conv[model]/Conv[21]/Conv2d[conv]/input.419
        self.module_217 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/Conv[model]/Conv[21]/torch.nn.BatchNorm2d2d[bn]/input.421
        self.module_218 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/Conv[model]/Conv[21]/LeakyReLU[act]/15153
        self.module_219 = torch.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[23]/Conv[cv1]/Conv2d[conv]/input.425
        self.module_220 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[23]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.427
        self.module_221 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[23]/Conv[cv1]/LeakyReLU[act]/input.429
        self.module_234 = torch.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[23]/Conv[cv2]/Conv2d[conv]/input.453
        self.module_235 = torch.nn.BatchNorm2d(num_features=384, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[23]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.455
        self.module_236 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[23]/Conv[cv2]/LeakyReLU[act]/15318
        self.module_237 = torch.nn.Conv2d(in_channels=768, out_channels=768, kernel_size=[1, 1], stride=[1, 1],
                                          padding=[0, 0], dilation=[1, 1], groups=1,
                                          bias=False)  # Model::Model/C3[model]/C3[23]/Conv[cv3]/Conv2d[conv]/input.459
        self.module_238 = torch.nn.BatchNorm2d(num_features=768, eps=0.001,
                                               momentum=0.03)  # Model::Model/C3[model]/C3[23]/Conv[cv3]/torch.nn.BatchNorm2d2d[bn]/input.461
        self.module_239 = torch.nn.LeakyReLU(negative_slope=0.1015625,
                                             inplace=True)  # Model::Model/C3[model]/C3[23]/Conv[cv3]/LeakyReLU[act]/input

        self.module_222 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.431
        self.module_223 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.433
        self.module_224 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.435
        self.module_225 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.437
        self.module_226 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.439
        self.module_227 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.441
        self.module_228 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.443
        self.module_229 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv1]/torch.nn.BatchNorm2d2d[bn]/input.445
        self.module_230 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.447
        self.module_231 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.449
        self.module_232 = torch.nn.BatchNorm2d(num_features=384, eps=0.001, momentum=0.03) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv2]/torch.nn.BatchNorm2d2d[bn]/input.451
        self.module_233 = torch.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #Model::Model/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/15291
        self.module_240 = torch.nn.Conv2d(in_channels=192, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[0]/15367
        self.module_241 = torch.nn.Conv2d(in_channels=384, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[1]/15386
        self.module_242 = torch.nn.Conv2d(in_channels=768, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[2]/15405
        self.stride = torch.tensor([8., 16., 32.])  #####
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light',
                      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                      'cow',
                      'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                      'frisbee',
                      'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                      'surfboard',
                      'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                      'apple',
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                      'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                      'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear',
                      'hair drier', 'toothbrush']
        self.add_244 = functional.Add()
        self.add_245 = functional.Add()
        self.add_246 = functional.Add()
        self.add_247 = functional.Add()
        self.add_248 = functional.Add()
        self.add_249 = functional.Add()
        self.add_250 = functional.Add()
        self.add_251 = functional.Add()
        self.add_252 = functional.Add()
        self.add_253 = functional.Add()
        self.add_254 = functional.Add()
        self.add_255 = functional.Add()
        self.add_256 = functional.Add()
        self.add_257 = functional.Add()

        self.cat_245 = functional.Cat()
        self.cat_249 = functional.Cat()
        self.cat_255 = functional.Cat()
        self.cat_257 = functional.Cat()
        self.cat_258 = functional.Cat()
        self.cat_259 = functional.Cat()
        self.cat_147 = functional.Cat()
        self.cat_260 = functional.Cat()
        self.cat_192 = functional.Cat()
        self.cat_195 = functional.Cat()
        self.cat_216 = functional.Cat()
        self.cat_219 = functional.Cat()
        self.cat_171 = functional.Cat()
    def forward(self, x):
        output_243 = x
        output_243 = self.module_0(output_243)
        output_243 = self.module_1(output_243)
        output_243 = self.module_2(output_243)
        output_243 = self.module_3(output_243)
        output_243 = self.module_4(output_243)
        output_243 = self.module_5(output_243)
        output_module_6 = self.module_6(output_243)
        output_module_6 = self.module_7(output_module_6)
        output_module_6 = self.module_8(output_module_6)
        output_module_9 = self.module_9(output_module_6)
        output_module_9 = self.module_10(output_module_9)
        output_module_9 = self.module_11(output_module_9)
        output_module_9 = self.module_12(output_module_9)
        output_module_9 = self.module_13(output_module_9)
        output_module_9 = self.module_14(output_module_9)
        output_244 = self.add_244(output_module_6,
                                  output_module_9)  # Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Add[skip_add]/input.31
        output_module_15 = self.module_15(output_244)
        output_module_15 = self.module_16(output_module_15)
        output_module_15 = self.module_17(output_module_15)
        output_module_15 = self.module_18(output_module_15)
        output_module_15 = self.module_19(output_module_15)
        output_module_15 = self.module_20(output_module_15)
        output_245 = self.add_245(output_244,
                                  output_module_15)  # Model::Model/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Add[skip_add]/13289
        output_module_21 = self.module_21(output_243)
        output_module_21 = self.module_22(output_module_21)
        output_module_21 = self.module_23(output_module_21)
        output_245 = self.cat_245((output_245, output_module_21),
                                  dim=1)  # Model::Model/C3[model]/C3[2]/Cat[cat]/input.47
        output_245 = self.module_24(output_245)
        output_245 = self.module_25(output_245)
        output_245 = self.module_26(output_245)
        output_245 = self.module_27(output_245)
        output_245 = self.module_28(output_245)
        output_245 = self.module_29(output_245)
        output_module_30 = self.module_30(output_245)
        output_module_30 = self.module_31(output_module_30)
        output_module_30 = self.module_32(output_module_30)
        output_module_33 = self.module_33(output_module_30)
        output_module_33 = self.module_34(output_module_33)
        output_module_33 = self.module_35(output_module_33)
        output_module_33 = self.module_36(output_module_33)
        output_module_33 = self.module_37(output_module_33)
        output_module_33 = self.module_38(output_module_33)
        output_246 = self.add_246(output_module_30,
                                  output_module_33)  # Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Add[skip_add]/input.77
        output_module_39 = self.module_39(output_246)
        output_module_39 = self.module_40(output_module_39)
        output_module_39 = self.module_41(output_module_39)
        output_module_39 = self.module_42(output_module_39)
        output_module_39 = self.module_43(output_module_39)
        output_module_39 = self.module_44(output_module_39)
        output_247 = self.add_247(output_246,
                                  output_module_39)  # Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Add[skip_add]/input.89
        output_module_45 = self.module_45(output_247)
        output_module_45 = self.module_46(output_module_45)
        output_module_45 = self.module_47(output_module_45)
        output_module_45 = self.module_48(output_module_45)
        output_module_45 = self.module_49(output_module_45)
        output_module_45 = self.module_50(output_module_45)
        output_248 = self.add_248(output_247,
                                  output_module_45)  # Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Add[skip_add]/input.101
        output_module_51 = self.module_51(output_248)
        output_module_51 = self.module_52(output_module_51)
        output_module_51 = self.module_53(output_module_51)
        output_module_51 = self.module_54(output_module_51)
        output_module_51 = self.module_55(output_module_51)
        output_module_51 = self.module_56(output_module_51)
        output_249 = self.add_249(output_248,
                                  output_module_51)  # Model::Model/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Add[skip_add]/13624
        output_module_57 = self.module_57(output_245)
        output_module_57 = self.module_58(output_module_57)
        output_module_57 = self.module_59(output_module_57)
        output_249 = self.cat_249((output_249, output_module_57),
                                  dim=1)  # Model::Model/C3[model]/C3[4]/Cat[cat]/input.117
        output_249 = self.module_60(output_249)
        output_249 = self.module_61(output_249)
        output_249 = self.module_62(output_249)
        output_module_63 = self.module_63(output_249)
        output_module_63 = self.module_64(output_module_63)
        output_module_63 = self.module_65(output_module_63)
        output_module_66 = self.module_66(output_module_63)
        output_module_66 = self.module_67(output_module_66)
        output_module_66 = self.module_68(output_module_66)
        output_module_69 = self.module_69(output_module_66)
        output_module_69 = self.module_70(output_module_69)
        output_module_69 = self.module_71(output_module_69)
        output_module_69 = self.module_72(output_module_69)
        output_module_69 = self.module_73(output_module_69)
        output_module_69 = self.module_74(output_module_69)
        output_250 = self.add_250(output_module_66,
                                  output_module_69)  # Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Add[skip_add]/input.147
        output_module_75 = self.module_75(output_250)
        output_module_75 = self.module_76(output_module_75)
        output_module_75 = self.module_77(output_module_75)
        output_module_75 = self.module_78(output_module_75)
        output_module_75 = self.module_79(output_module_75)
        output_module_75 = self.module_80(output_module_75)
        output_251 = self.add_251(output_250,
                                  output_module_75)  # Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Add[skip_add]/input.159
        output_module_81 = self.module_81(output_251)
        output_module_81 = self.module_82(output_module_81)
        output_module_81 = self.module_83(output_module_81)
        output_module_81 = self.module_84(output_module_81)
        output_module_81 = self.module_85(output_module_81)
        output_module_81 = self.module_86(output_module_81)
        output_252 = self.add_252(output_251,
                                  output_module_81)  # Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Add[skip_add]/input.171
        output_module_87 = self.module_87(output_252)
        output_module_87 = self.module_88(output_module_87)
        output_module_87 = self.module_89(output_module_87)
        output_module_87 = self.module_90(output_module_87)
        output_module_87 = self.module_91(output_module_87)
        output_module_87 = self.module_92(output_module_87)
        output_253 = self.add_253(output_252,
                                  output_module_87)  # Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Add[skip_add]/input.183
        output_module_93 = self.module_93(output_253)
        output_module_93 = self.module_94(output_module_93)
        output_module_93 = self.module_95(output_module_93)
        output_module_93 = self.module_96(output_module_93)
        output_module_93 = self.module_97(output_module_93)
        output_module_93 = self.module_98(output_module_93)
        output_254 = self.add_254(output_253,
                                  output_module_93)  # Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Add[skip_add]/input.195
        output_module_99 = self.module_99(output_254)
        output_module_99 = self.module_100(output_module_99)
        output_module_99 = self.module_101(output_module_99)
        output_module_99 = self.module_102(output_module_99)
        output_module_99 = self.module_103(output_module_99)
        output_module_99 = self.module_104(output_module_99)
        output_255 = self.add_255(output_254,
                                  output_module_99)  # Model::Model/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Add[skip_add]/14071
        output_module_105 = self.module_105(output_module_63)
        output_module_105 = self.module_106(output_module_105)
        output_module_105 = self.module_107(output_module_105)
        output_255 = self.cat_255((output_255, output_module_105),
                                  dim=1)  # Model::Model/C3[model]/C3[6]/Cat[cat]/input.211
        output_255 = self.module_108(output_255)
        output_255 = self.module_109(output_255)
        output_255 = self.module_110(output_255)
        output_module_111 = self.module_111(output_255)
        output_module_111 = self.module_112(output_module_111)
        output_module_111 = self.module_113(output_module_111)
        output_module_114 = self.module_114(output_module_111)
        output_module_114 = self.module_115(output_module_114)
        output_module_114 = self.module_116(output_module_114)
        output_module_117 = self.module_117(output_module_114)
        output_module_117 = self.module_118(output_module_117)
        output_module_117 = self.module_119(output_module_117)
        output_module_117 = self.module_120(output_module_117)
        output_module_117 = self.module_121(output_module_117)
        output_module_117 = self.module_122(output_module_117)
        output_256 = self.add_256(output_module_114,
                                  output_module_117)  # Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Add[skip_add]/input.241
        output_module_123 = self.module_123(output_256)
        output_module_123 = self.module_124(output_module_123)
        output_module_123 = self.module_125(output_module_123)
        output_module_123 = self.module_126(output_module_123)
        output_module_123 = self.module_127(output_module_123)
        output_module_123 = self.module_128(output_module_123)
        output_257 = self.add_257(output_256,
                                  output_module_123)  # Model::Model/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Add[skip_add]/14294
        output_module_129 = self.module_129(output_module_111)
        output_module_129 = self.module_130(output_module_129)
        output_module_129 = self.module_131(output_module_129)
        output_257 = self.cat_257((output_257, output_module_129),
                                  dim=1)  # Model::Model/C3[model]/C3[8]/Cat[cat]/input.257
        output_257 = self.module_132(output_257)
        output_257 = self.module_133(output_257)
        output_257 = self.module_134(output_257)
        output_257 = self.module_135(output_257)
        output_257 = self.module_136(output_257)
        output_257 = self.module_137(output_257)
        output_module_138 = self.module_138(output_257)
        output_module_139 = self.module_139(output_module_138)
        output_module_140 = self.module_140(output_module_139)
        output_258 = self.cat_258((output_257, output_module_138, output_module_139,
                                   output_module_140), dim=1)  # Model::Model/SPPF[model]/SPPF[9]/Cat[cat]/input.269
        output_258 = self.module_141(output_258)
        output_258 = self.module_142(output_258)
        output_258 = self.module_143(output_258)
        output_258 = self.module_144(output_258)
        output_258 = self.module_145(output_258)
        output_258 = self.module_146(output_258)
        output_259 = torch.nn.functional.interpolate(input=output_258, size=None, scale_factor=[2.0, 2.0],
                                                     mode='nearest')  # Model::Model/Upsample[model]/Upsample[11]/14482
        output_259 = self.cat_259((output_259, output_255),
                                  dim=1)  # Model::Model/Concat[model]/Concat[12]/Cat[cat]/input.283
        output_module_147 = self.module_147(output_259)
        output_module_147 = self.module_148(output_module_147)
        output_module_147 = self.module_149(output_module_147)
        output_module_147 = self.module_150(output_module_147)
        output_module_147 = self.module_151(output_module_147)
        output_module_147 = self.module_152(output_module_147)
        output_module_147 = self.module_153(output_module_147)
        output_module_147 = self.module_154(output_module_147)
        output_module_147 = self.module_155(output_module_147)
        output_module_147 = self.module_156(output_module_147)
        output_module_147 = self.module_157(output_module_147)
        output_module_147 = self.module_158(output_module_147)
        output_module_147 = self.module_159(output_module_147)
        output_module_147 = self.module_160(output_module_147)
        output_module_147 = self.module_161(output_module_147)
        output_module_162 = self.module_162(output_259)
        output_module_162 = self.module_163(output_module_162)
        output_module_162 = self.module_164(output_module_162)
        output_module_147 = self.cat_147((output_module_147, output_module_162),
                                         dim=1)  # Model::Model/C3[model]/C3[13]/Cat[cat]/input.317
        output_module_147 = self.module_165(output_module_147)
        output_module_147 = self.module_166(output_module_147)
        output_module_147 = self.module_167(output_module_147)
        output_module_147 = self.module_168(output_module_147)
        output_module_147 = self.module_169(output_module_147)
        output_module_147 = self.module_170(output_module_147)
        output_260 = torch.nn.functional.interpolate(input=output_module_147, size=None, scale_factor=[2.0, 2.0],
                                                     mode='nearest')  # Model::Model/Upsample[model]/Upsample[15]/14709
        output_260 = self.cat_260((output_260, output_249),
                                  dim=1)  # Model::Model/Concat[model]/Concat[16]/Cat[cat]/input.331
        output_module_171 = self.module_171(output_260)
        output_module_171 = self.module_172(output_module_171)
        output_module_171 = self.module_173(output_module_171)
        output_module_171 = self.module_174(output_module_171)
        output_module_171 = self.module_175(output_module_171)
        output_module_171 = self.module_176(output_module_171)
        output_module_171 = self.module_177(output_module_171)
        output_module_171 = self.module_178(output_module_171)
        output_module_171 = self.module_179(output_module_171)
        output_module_171 = self.module_180(output_module_171)
        output_module_171 = self.module_181(output_module_171)
        output_module_171 = self.module_182(output_module_171)
        output_module_171 = self.module_183(output_module_171)
        output_module_171 = self.module_184(output_module_171)
        output_module_171 = self.module_185(output_module_171)
        output_module_186 = self.module_186(output_260)
        output_module_186 = self.module_187(output_module_186)
        output_module_186 = self.module_188(output_module_186)
        output_module_171 = self.cat_171((output_module_171, output_module_186),
                                         dim=1)  # Model::Model/C3[model]/C3[17]/Cat[cat]/input.365
        output_module_171 = self.module_189(output_module_171)
        output_module_171 = self.module_190(output_module_171)
        output_module_171 = self.module_191(output_module_171)
        output_module_192 = self.module_192(output_module_171)
        output_module_192 = self.module_193(output_module_192)
        output_module_192 = self.module_194(output_module_192)
        output_module_192 = self.cat_192((output_module_192, output_module_147),
                                         dim=1)  # Model::Model/Concat[model]/Concat[19]/Cat[cat]/input.377
        output_module_195 = self.module_195(output_module_192)
        output_module_195 = self.module_196(output_module_195)
        output_module_195 = self.module_197(output_module_195)
        output_module_195 = self.module_198(output_module_195)
        output_module_195 = self.module_199(output_module_195)
        output_module_195 = self.module_200(output_module_195)
        output_module_195 = self.module_201(output_module_195)
        output_module_195 = self.module_202(output_module_195)
        output_module_195 = self.module_203(output_module_195)
        output_module_195 = self.module_204(output_module_195)
        output_module_195 = self.module_205(output_module_195)
        output_module_195 = self.module_206(output_module_195)
        output_module_195 = self.module_207(output_module_195)
        output_module_195 = self.module_208(output_module_195)
        output_module_195 = self.module_209(output_module_195)
        output_module_210 = self.module_210(output_module_192)
        output_module_210 = self.module_211(output_module_210)
        output_module_210 = self.module_212(output_module_210)
        output_module_195 = self.cat_195((output_module_195, output_module_210),
                                         dim=1)  # Model::Model/C3[model]/C3[20]/Cat[cat]/input.411
        output_module_195 = self.module_213(output_module_195)
        output_module_195 = self.module_214(output_module_195)
        output_module_195 = self.module_215(output_module_195)
        output_module_216 = self.module_216(output_module_195)
        output_module_216 = self.module_217(output_module_216)
        output_module_216 = self.module_218(output_module_216)
        output_module_216 = self.cat_216((output_module_216, output_258),
                                         dim=1)  # Model::Model/Concat[model]/Concat[22]/Cat[cat]/input.423
        output_module_219 = self.module_219(output_module_216)
        output_module_219 = self.module_220(output_module_219)
        output_module_219 = self.module_221(output_module_219)
        output_module_219 = self.module_222(output_module_219)
        output_module_219 = self.module_223(output_module_219)
        output_module_219 = self.module_224(output_module_219)
        output_module_219 = self.module_225(output_module_219)
        output_module_219 = self.module_226(output_module_219)
        output_module_219 = self.module_227(output_module_219)
        output_module_219 = self.module_228(output_module_219)
        output_module_219 = self.module_229(output_module_219)
        output_module_219 = self.module_230(output_module_219)
        output_module_219 = self.module_231(output_module_219)
        output_module_219 = self.module_232(output_module_219)
        output_module_219 = self.module_233(output_module_219)
        output_module_234 = self.module_234(output_module_216)
        output_module_234 = self.module_235(output_module_234)
        output_module_234 = self.module_236(output_module_234)
        output_module_219 = self.cat_219((output_module_219, output_module_234),
                                         dim=1)  # Model::Model/C3[model]/C3[23]/Cat[cat]/input.457
        output_module_219 = self.module_237(output_module_219)
        output_module_219 = self.module_238(output_module_219)
        output_module_219 = self.module_239(output_module_219)
        output_module_240 = self.module_240(output_module_171)
        output_module_241 = self.module_241(output_module_195)
        output_module_219 = self.module_242(output_module_219)
        return output_module_240, output_module_241, output_module_219



