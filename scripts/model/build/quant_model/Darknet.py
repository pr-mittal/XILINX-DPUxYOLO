# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class Darknet(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(Darknet, self).__init__()
        self.module_0 = py_nndct.nn.Module('nndct_const') #Darknet::11372
        self.module_1 = py_nndct.nn.Module('nndct_const') #Darknet::11376
        self.module_2 = py_nndct.nn.Module('nndct_const') #Darknet::11380
        self.module_3 = py_nndct.nn.Module('nndct_const') #Darknet::11384
        self.module_4 = py_nndct.nn.Module('nndct_const') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10202
        self.module_5 = py_nndct.nn.Module('nndct_const') #Darknet::Darknet/10297
        self.module_6 = py_nndct.nn.Module('nndct_const') #Darknet::Darknet/10361
        self.module_7 = py_nndct.nn.Module('nndct_const') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10593
        self.module_8 = py_nndct.nn.Module('nndct_const') #Darknet::Darknet/10688
        self.module_9 = py_nndct.nn.Module('nndct_const') #Darknet::Darknet/10752
        self.module_10 = py_nndct.nn.Module('nndct_const') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10984
        self.module_11 = py_nndct.nn.Input() #Darknet::input_0
        self.module_12 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/8535
        self.module_13 = py_nndct.nn.Module('nndct_tensor') #Darknet::Darknet/8536
        self.module_14 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[0]/Conv2d[conv_0]/input.3
        self.module_15 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[0]/LeakyReLU[leaky_0]/input.7
        self.module_16 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[1]/Conv2d[conv_1]/input.9
        self.module_17 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[1]/LeakyReLU[leaky_1]/input.13
        self.module_18 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[2]/Conv2d[conv_2]/input.15
        self.module_19 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[2]/LeakyReLU[leaky_2]/input.19
        self.module_20 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[3]/Conv2d[conv_3]/input.21
        self.module_21 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[3]/LeakyReLU[leaky_3]/8644
        self.module_22 = py_nndct.nn.Add() #Darknet::Darknet/input.25
        self.module_23 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[5]/Conv2d[conv_5]/input.27
        self.module_24 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[5]/LeakyReLU[leaky_5]/input.31
        self.module_25 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[6]/Conv2d[conv_6]/input.33
        self.module_26 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[6]/LeakyReLU[leaky_6]/input.37
        self.module_27 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[7]/Conv2d[conv_7]/input.39
        self.module_28 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[7]/LeakyReLU[leaky_7]/8727
        self.module_29 = py_nndct.nn.Add() #Darknet::Darknet/input.43
        self.module_30 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[9]/Conv2d[conv_9]/input.45
        self.module_31 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[9]/LeakyReLU[leaky_9]/input.49
        self.module_32 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[10]/Conv2d[conv_10]/input.51
        self.module_33 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[10]/LeakyReLU[leaky_10]/8783
        self.module_34 = py_nndct.nn.Add() #Darknet::Darknet/input.55
        self.module_35 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[12]/Conv2d[conv_12]/input.57
        self.module_36 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[12]/LeakyReLU[leaky_12]/input.61
        self.module_37 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[13]/Conv2d[conv_13]/input.63
        self.module_38 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[13]/LeakyReLU[leaky_13]/input.67
        self.module_39 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[14]/Conv2d[conv_14]/input.69
        self.module_40 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[14]/LeakyReLU[leaky_14]/8866
        self.module_41 = py_nndct.nn.Add() #Darknet::Darknet/input.73
        self.module_42 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[16]/Conv2d[conv_16]/input.75
        self.module_43 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[16]/LeakyReLU[leaky_16]/input.79
        self.module_44 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[17]/Conv2d[conv_17]/input.81
        self.module_45 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[17]/LeakyReLU[leaky_17]/8922
        self.module_46 = py_nndct.nn.Add() #Darknet::Darknet/input.85
        self.module_47 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[19]/Conv2d[conv_19]/input.87
        self.module_48 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[19]/LeakyReLU[leaky_19]/input.91
        self.module_49 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[20]/Conv2d[conv_20]/input.93
        self.module_50 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[20]/LeakyReLU[leaky_20]/8978
        self.module_51 = py_nndct.nn.Add() #Darknet::Darknet/input.97
        self.module_52 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[22]/Conv2d[conv_22]/input.99
        self.module_53 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[22]/LeakyReLU[leaky_22]/input.103
        self.module_54 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[23]/Conv2d[conv_23]/input.105
        self.module_55 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[23]/LeakyReLU[leaky_23]/9034
        self.module_56 = py_nndct.nn.Add() #Darknet::Darknet/input.109
        self.module_57 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[25]/Conv2d[conv_25]/input.111
        self.module_58 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[25]/LeakyReLU[leaky_25]/input.115
        self.module_59 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[26]/Conv2d[conv_26]/input.117
        self.module_60 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[26]/LeakyReLU[leaky_26]/9090
        self.module_61 = py_nndct.nn.Add() #Darknet::Darknet/input.121
        self.module_62 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[28]/Conv2d[conv_28]/input.123
        self.module_63 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[28]/LeakyReLU[leaky_28]/input.127
        self.module_64 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[29]/Conv2d[conv_29]/input.129
        self.module_65 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[29]/LeakyReLU[leaky_29]/9146
        self.module_66 = py_nndct.nn.Add() #Darknet::Darknet/input.133
        self.module_67 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[31]/Conv2d[conv_31]/input.135
        self.module_68 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[31]/LeakyReLU[leaky_31]/input.139
        self.module_69 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[32]/Conv2d[conv_32]/input.141
        self.module_70 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[32]/LeakyReLU[leaky_32]/9202
        self.module_71 = py_nndct.nn.Add() #Darknet::Darknet/input.145
        self.module_72 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[34]/Conv2d[conv_34]/input.147
        self.module_73 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[34]/LeakyReLU[leaky_34]/input.151
        self.module_74 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[35]/Conv2d[conv_35]/input.153
        self.module_75 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[35]/LeakyReLU[leaky_35]/9258
        self.module_76 = py_nndct.nn.Add() #Darknet::Darknet/input.157
        self.module_77 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[37]/Conv2d[conv_37]/input.159
        self.module_78 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[37]/LeakyReLU[leaky_37]/input.163
        self.module_79 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[38]/Conv2d[conv_38]/input.165
        self.module_80 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[38]/LeakyReLU[leaky_38]/input.169
        self.module_81 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[39]/Conv2d[conv_39]/input.171
        self.module_82 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[39]/LeakyReLU[leaky_39]/9341
        self.module_83 = py_nndct.nn.Add() #Darknet::Darknet/input.175
        self.module_84 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[41]/Conv2d[conv_41]/input.177
        self.module_85 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[41]/LeakyReLU[leaky_41]/input.181
        self.module_86 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[42]/Conv2d[conv_42]/input.183
        self.module_87 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[42]/LeakyReLU[leaky_42]/9397
        self.module_88 = py_nndct.nn.Add() #Darknet::Darknet/input.187
        self.module_89 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[44]/Conv2d[conv_44]/input.189
        self.module_90 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[44]/LeakyReLU[leaky_44]/input.193
        self.module_91 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[45]/Conv2d[conv_45]/input.195
        self.module_92 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[45]/LeakyReLU[leaky_45]/9453
        self.module_93 = py_nndct.nn.Add() #Darknet::Darknet/input.199
        self.module_94 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[47]/Conv2d[conv_47]/input.201
        self.module_95 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[47]/LeakyReLU[leaky_47]/input.205
        self.module_96 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[48]/Conv2d[conv_48]/input.207
        self.module_97 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[48]/LeakyReLU[leaky_48]/9509
        self.module_98 = py_nndct.nn.Add() #Darknet::Darknet/input.211
        self.module_99 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[50]/Conv2d[conv_50]/input.213
        self.module_100 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[50]/LeakyReLU[leaky_50]/input.217
        self.module_101 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[51]/Conv2d[conv_51]/input.219
        self.module_102 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[51]/LeakyReLU[leaky_51]/9565
        self.module_103 = py_nndct.nn.Add() #Darknet::Darknet/input.223
        self.module_104 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[53]/Conv2d[conv_53]/input.225
        self.module_105 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[53]/LeakyReLU[leaky_53]/input.229
        self.module_106 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[54]/Conv2d[conv_54]/input.231
        self.module_107 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[54]/LeakyReLU[leaky_54]/9621
        self.module_108 = py_nndct.nn.Add() #Darknet::Darknet/input.235
        self.module_109 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[56]/Conv2d[conv_56]/input.237
        self.module_110 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[56]/LeakyReLU[leaky_56]/input.241
        self.module_111 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[57]/Conv2d[conv_57]/input.243
        self.module_112 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[57]/LeakyReLU[leaky_57]/9677
        self.module_113 = py_nndct.nn.Add() #Darknet::Darknet/input.247
        self.module_114 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[59]/Conv2d[conv_59]/input.249
        self.module_115 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[59]/LeakyReLU[leaky_59]/input.253
        self.module_116 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[60]/Conv2d[conv_60]/input.255
        self.module_117 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[60]/LeakyReLU[leaky_60]/9733
        self.module_118 = py_nndct.nn.Add() #Darknet::Darknet/input.259
        self.module_119 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[62]/Conv2d[conv_62]/input.261
        self.module_120 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[62]/LeakyReLU[leaky_62]/input.265
        self.module_121 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[63]/Conv2d[conv_63]/input.267
        self.module_122 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[63]/LeakyReLU[leaky_63]/input.271
        self.module_123 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[64]/Conv2d[conv_64]/input.273
        self.module_124 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[64]/LeakyReLU[leaky_64]/9816
        self.module_125 = py_nndct.nn.Add() #Darknet::Darknet/input.277
        self.module_126 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[66]/Conv2d[conv_66]/input.279
        self.module_127 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[66]/LeakyReLU[leaky_66]/input.283
        self.module_128 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[67]/Conv2d[conv_67]/input.285
        self.module_129 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[67]/LeakyReLU[leaky_67]/9872
        self.module_130 = py_nndct.nn.Add() #Darknet::Darknet/input.289
        self.module_131 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[69]/Conv2d[conv_69]/input.291
        self.module_132 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[69]/LeakyReLU[leaky_69]/input.295
        self.module_133 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[70]/Conv2d[conv_70]/input.297
        self.module_134 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[70]/LeakyReLU[leaky_70]/9928
        self.module_135 = py_nndct.nn.Add() #Darknet::Darknet/input.301
        self.module_136 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[72]/Conv2d[conv_72]/input.303
        self.module_137 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[72]/LeakyReLU[leaky_72]/input.307
        self.module_138 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[73]/Conv2d[conv_73]/input.309
        self.module_139 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[73]/LeakyReLU[leaky_73]/9984
        self.module_140 = py_nndct.nn.Add() #Darknet::Darknet/input.313
        self.module_141 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[75]/Conv2d[conv_75]/input.315
        self.module_142 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[75]/LeakyReLU[leaky_75]/input.319
        self.module_143 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[76]/Conv2d[conv_76]/input.321
        self.module_144 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[76]/LeakyReLU[leaky_76]/input.325
        self.module_145 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[77]/Conv2d[conv_77]/input.327
        self.module_146 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[77]/LeakyReLU[leaky_77]/input.331
        self.module_147 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[78]/Conv2d[conv_78]/input.333
        self.module_148 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[78]/LeakyReLU[leaky_78]/input.337
        self.module_149 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[79]/Conv2d[conv_79]/input.339
        self.module_150 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[79]/LeakyReLU[leaky_79]/input.343
        self.module_151 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[80]/Conv2d[conv_80]/input.345
        self.module_152 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[80]/LeakyReLU[leaky_80]/input.349
        self.module_153 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[81]/Conv2d[conv_81]/10167
        self.module_154 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10169
        self.module_155 = py_nndct.nn.Module('nndct_tensor') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10170
        self.module_156 = py_nndct.nn.Module('aten::div') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10172
        self.module_157 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10174
        self.module_158 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10182
        self.module_159 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10186
        self.module_160 = py_nndct.nn.Module('nndct_reshape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10192
        self.module_161 = py_nndct.nn.Module('nndct_permute') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10199
        self.module_162 = py_nndct.nn.Module('nndct_contiguous') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10201
        self.module_163 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10239
        self.module_164 = py_nndct.nn.Sigmoid() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10240
        self.module_165 = py_nndct.nn.Add() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10242
        self.module_166 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10243
        self.module_167 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10255
        self.module_168 = py_nndct.nn.Module('nndct_elemwise_exp') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10256
        self.module_169 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10257
        self.module_170 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10269
        self.module_171 = py_nndct.nn.Sigmoid() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10270
        self.module_172 = py_nndct.nn.Module('nndct_reshape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[82]/YOLOLayer[yolo_82]/10281
        self.module_173 = py_nndct.nn.Cat() #Darknet::Darknet/10284
        self.module_174 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/10289
        self.module_175 = py_nndct.nn.Module('nndct_tensor') #Darknet::Darknet/10290
        self.module_176 = py_nndct.nn.Module('aten::div') #Darknet::Darknet/10299
        self.module_177 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/11373
        self.module_178 = py_nndct.nn.Int() #Darknet::Darknet/10302
        self.module_179 = py_nndct.nn.Int() #Darknet::Darknet/10305
        self.module_180 = py_nndct.nn.strided_slice() #Darknet::Darknet/10310
        self.module_181 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[84]/Conv2d[conv_84]/input.353
        self.module_182 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[84]/LeakyReLU[leaky_84]/x.1
        self.module_183 = py_nndct.nn.Interpolate() #Darknet::Darknet/Sequential[module_list]/ModuleList[85]/Upsample[upsample_85]/10345
        self.module_184 = py_nndct.nn.Cat() #Darknet::Darknet/10348
        self.module_185 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/10353
        self.module_186 = py_nndct.nn.Module('nndct_tensor') #Darknet::Darknet/10354
        self.module_187 = py_nndct.nn.Module('aten::div') #Darknet::Darknet/10363
        self.module_188 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/11377
        self.module_189 = py_nndct.nn.Int() #Darknet::Darknet/10366
        self.module_190 = py_nndct.nn.Int() #Darknet::Darknet/10369
        self.module_191 = py_nndct.nn.strided_slice() #Darknet::Darknet/10374
        self.module_192 = py_nndct.nn.Conv2d(in_channels=768, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[87]/Conv2d[conv_87]/input.359
        self.module_193 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[87]/LeakyReLU[leaky_87]/input.363
        self.module_194 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[88]/Conv2d[conv_88]/input.365
        self.module_195 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[88]/LeakyReLU[leaky_88]/input.369
        self.module_196 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[89]/Conv2d[conv_89]/input.371
        self.module_197 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[89]/LeakyReLU[leaky_89]/input.375
        self.module_198 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[90]/Conv2d[conv_90]/input.377
        self.module_199 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[90]/LeakyReLU[leaky_90]/input.381
        self.module_200 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[91]/Conv2d[conv_91]/input.383
        self.module_201 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[91]/LeakyReLU[leaky_91]/input.387
        self.module_202 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[92]/Conv2d[conv_92]/input.389
        self.module_203 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[92]/LeakyReLU[leaky_92]/input.393
        self.module_204 = py_nndct.nn.Conv2d(in_channels=512, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[93]/Conv2d[conv_93]/10558
        self.module_205 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10560
        self.module_206 = py_nndct.nn.Module('nndct_tensor') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10561
        self.module_207 = py_nndct.nn.Module('aten::div') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10563
        self.module_208 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10565
        self.module_209 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10573
        self.module_210 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10577
        self.module_211 = py_nndct.nn.Module('nndct_reshape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10583
        self.module_212 = py_nndct.nn.Module('nndct_permute') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10590
        self.module_213 = py_nndct.nn.Module('nndct_contiguous') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10592
        self.module_214 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10630
        self.module_215 = py_nndct.nn.Sigmoid() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10631
        self.module_216 = py_nndct.nn.Add() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10633
        self.module_217 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10634
        self.module_218 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10646
        self.module_219 = py_nndct.nn.Module('nndct_elemwise_exp') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10647
        self.module_220 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10648
        self.module_221 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10660
        self.module_222 = py_nndct.nn.Sigmoid() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10661
        self.module_223 = py_nndct.nn.Module('nndct_reshape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[94]/YOLOLayer[yolo_94]/10672
        self.module_224 = py_nndct.nn.Cat() #Darknet::Darknet/10675
        self.module_225 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/10680
        self.module_226 = py_nndct.nn.Module('nndct_tensor') #Darknet::Darknet/10681
        self.module_227 = py_nndct.nn.Module('aten::div') #Darknet::Darknet/10690
        self.module_228 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/11381
        self.module_229 = py_nndct.nn.Int() #Darknet::Darknet/10693
        self.module_230 = py_nndct.nn.Int() #Darknet::Darknet/10696
        self.module_231 = py_nndct.nn.strided_slice() #Darknet::Darknet/10701
        self.module_232 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[96]/Conv2d[conv_96]/input.397
        self.module_233 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[96]/LeakyReLU[leaky_96]/x
        self.module_234 = py_nndct.nn.Interpolate() #Darknet::Darknet/Sequential[module_list]/ModuleList[97]/Upsample[upsample_97]/10736
        self.module_235 = py_nndct.nn.Cat() #Darknet::Darknet/10739
        self.module_236 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/10744
        self.module_237 = py_nndct.nn.Module('nndct_tensor') #Darknet::Darknet/10745
        self.module_238 = py_nndct.nn.Module('aten::div') #Darknet::Darknet/10754
        self.module_239 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/11385
        self.module_240 = py_nndct.nn.Int() #Darknet::Darknet/10757
        self.module_241 = py_nndct.nn.Int() #Darknet::Darknet/10760
        self.module_242 = py_nndct.nn.strided_slice() #Darknet::Darknet/10765
        self.module_243 = py_nndct.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[99]/Conv2d[conv_99]/input.403
        self.module_244 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[99]/LeakyReLU[leaky_99]/input.407
        self.module_245 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[100]/Conv2d[conv_100]/input.409
        self.module_246 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[100]/LeakyReLU[leaky_100]/input.413
        self.module_247 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[101]/Conv2d[conv_101]/input.415
        self.module_248 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[101]/LeakyReLU[leaky_101]/input.419
        self.module_249 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[102]/Conv2d[conv_102]/input.421
        self.module_250 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[102]/LeakyReLU[leaky_102]/input.425
        self.module_251 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[103]/Conv2d[conv_103]/input.427
        self.module_252 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[103]/LeakyReLU[leaky_103]/input.431
        self.module_253 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[104]/Conv2d[conv_104]/input.433
        self.module_254 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=False) #Darknet::Darknet/Sequential[module_list]/ModuleList[104]/LeakyReLU[leaky_104]/input
        self.module_255 = py_nndct.nn.Conv2d(in_channels=256, out_channels=255, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Darknet::Darknet/Sequential[module_list]/ModuleList[105]/Conv2d[conv_105]/10949
        self.module_256 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10951
        self.module_257 = py_nndct.nn.Module('nndct_tensor') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10952
        self.module_258 = py_nndct.nn.Module('aten::div') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10954
        self.module_259 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10956
        self.module_260 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10964
        self.module_261 = py_nndct.nn.Module('nndct_shape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10968
        self.module_262 = py_nndct.nn.Module('nndct_reshape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10974
        self.module_263 = py_nndct.nn.Module('nndct_permute') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10981
        self.module_264 = py_nndct.nn.Module('nndct_contiguous') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/10983
        self.module_265 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11021
        self.module_266 = py_nndct.nn.Sigmoid() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11022
        self.module_267 = py_nndct.nn.Add() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11024
        self.module_268 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11025
        self.module_269 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11037
        self.module_270 = py_nndct.nn.Module('nndct_elemwise_exp') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11038
        self.module_271 = py_nndct.nn.Module('nndct_elemwise_mul') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11039
        self.module_272 = py_nndct.nn.strided_slice() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11051
        self.module_273 = py_nndct.nn.Sigmoid() #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11052
        self.module_274 = py_nndct.nn.Module('nndct_reshape') #Darknet::Darknet/YOLOLayer[module_list]/ModuleList[106]/YOLOLayer[yolo_106]/11063
        self.module_275 = py_nndct.nn.Cat() #Darknet::Darknet/11066
        self.module_list_82_yolo_82_anchor_grid = torch.nn.parameter.Parameter(torch.Tensor(1, 3, 1, 1, 2))
        self.module_list_94_yolo_94_anchor_grid = torch.nn.parameter.Parameter(torch.Tensor(1, 3, 1, 1, 2))
        self.module_list_106_yolo_106_anchor_grid = torch.nn.parameter.Parameter(torch.Tensor(1, 3, 1, 1, 2))

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(data=0, dtype=torch.int64, device='cpu')
        output_module_1 = self.module_1(data=0, dtype=torch.int64, device='cpu')
        output_module_2 = self.module_2(data=0, dtype=torch.int64, device='cpu')
        output_module_3 = self.module_3(data=0, dtype=torch.int64, device='cpu')
        output_module_4 = self.module_4(data=[[[[[0.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0],[4.0,0.0],[5.0,0.0],[6.0,0.0],[7.0,0.0],[8.0,0.0],[9.0,0.0],[10.0,0.0],[11.0,0.0],[12.0,0.0]],[[0.0,1.0],[1.0,1.0],[2.0,1.0],[3.0,1.0],[4.0,1.0],[5.0,1.0],[6.0,1.0],[7.0,1.0],[8.0,1.0],[9.0,1.0],[10.0,1.0],[11.0,1.0],[12.0,1.0]],[[0.0,2.0],[1.0,2.0],[2.0,2.0],[3.0,2.0],[4.0,2.0],[5.0,2.0],[6.0,2.0],[7.0,2.0],[8.0,2.0],[9.0,2.0],[10.0,2.0],[11.0,2.0],[12.0,2.0]],[[0.0,3.0],[1.0,3.0],[2.0,3.0],[3.0,3.0],[4.0,3.0],[5.0,3.0],[6.0,3.0],[7.0,3.0],[8.0,3.0],[9.0,3.0],[10.0,3.0],[11.0,3.0],[12.0,3.0]],[[0.0,4.0],[1.0,4.0],[2.0,4.0],[3.0,4.0],[4.0,4.0],[5.0,4.0],[6.0,4.0],[7.0,4.0],[8.0,4.0],[9.0,4.0],[10.0,4.0],[11.0,4.0],[12.0,4.0]],[[0.0,5.0],[1.0,5.0],[2.0,5.0],[3.0,5.0],[4.0,5.0],[5.0,5.0],[6.0,5.0],[7.0,5.0],[8.0,5.0],[9.0,5.0],[10.0,5.0],[11.0,5.0],[12.0,5.0]],[[0.0,6.0],[1.0,6.0],[2.0,6.0],[3.0,6.0],[4.0,6.0],[5.0,6.0],[6.0,6.0],[7.0,6.0],[8.0,6.0],[9.0,6.0],[10.0,6.0],[11.0,6.0],[12.0,6.0]],[[0.0,7.0],[1.0,7.0],[2.0,7.0],[3.0,7.0],[4.0,7.0],[5.0,7.0],[6.0,7.0],[7.0,7.0],[8.0,7.0],[9.0,7.0],[10.0,7.0],[11.0,7.0],[12.0,7.0]],[[0.0,8.0],[1.0,8.0],[2.0,8.0],[3.0,8.0],[4.0,8.0],[5.0,8.0],[6.0,8.0],[7.0,8.0],[8.0,8.0],[9.0,8.0],[10.0,8.0],[11.0,8.0],[12.0,8.0]],[[0.0,9.0],[1.0,9.0],[2.0,9.0],[3.0,9.0],[4.0,9.0],[5.0,9.0],[6.0,9.0],[7.0,9.0],[8.0,9.0],[9.0,9.0],[10.0,9.0],[11.0,9.0],[12.0,9.0]],[[0.0,10.0],[1.0,10.0],[2.0,10.0],[3.0,10.0],[4.0,10.0],[5.0,10.0],[6.0,10.0],[7.0,10.0],[8.0,10.0],[9.0,10.0],[10.0,10.0],[11.0,10.0],[12.0,10.0]],[[0.0,11.0],[1.0,11.0],[2.0,11.0],[3.0,11.0],[4.0,11.0],[5.0,11.0],[6.0,11.0],[7.0,11.0],[8.0,11.0],[9.0,11.0],[10.0,11.0],[11.0,11.0],[12.0,11.0]],[[0.0,12.0],[1.0,12.0],[2.0,12.0],[3.0,12.0],[4.0,12.0],[5.0,12.0],[6.0,12.0],[7.0,12.0],[8.0,12.0],[9.0,12.0],[10.0,12.0],[11.0,12.0],[12.0,12.0]]]]], dtype=torch.float, device='cpu')
        output_module_5 = self.module_5(data=1, dtype=torch.int64, device='cpu')
        output_module_6 = self.module_6(data=1, dtype=torch.int64, device='cpu')
        output_module_7 = self.module_7(data=[[[[[0.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0],[4.0,0.0],[5.0,0.0],[6.0,0.0],[7.0,0.0],[8.0,0.0],[9.0,0.0],[10.0,0.0],[11.0,0.0],[12.0,0.0],[13.0,0.0],[14.0,0.0],[15.0,0.0],[16.0,0.0],[17.0,0.0],[18.0,0.0],[19.0,0.0],[20.0,0.0],[21.0,0.0],[22.0,0.0],[23.0,0.0],[24.0,0.0],[25.0,0.0]],[[0.0,1.0],[1.0,1.0],[2.0,1.0],[3.0,1.0],[4.0,1.0],[5.0,1.0],[6.0,1.0],[7.0,1.0],[8.0,1.0],[9.0,1.0],[10.0,1.0],[11.0,1.0],[12.0,1.0],[13.0,1.0],[14.0,1.0],[15.0,1.0],[16.0,1.0],[17.0,1.0],[18.0,1.0],[19.0,1.0],[20.0,1.0],[21.0,1.0],[22.0,1.0],[23.0,1.0],[24.0,1.0],[25.0,1.0]],[[0.0,2.0],[1.0,2.0],[2.0,2.0],[3.0,2.0],[4.0,2.0],[5.0,2.0],[6.0,2.0],[7.0,2.0],[8.0,2.0],[9.0,2.0],[10.0,2.0],[11.0,2.0],[12.0,2.0],[13.0,2.0],[14.0,2.0],[15.0,2.0],[16.0,2.0],[17.0,2.0],[18.0,2.0],[19.0,2.0],[20.0,2.0],[21.0,2.0],[22.0,2.0],[23.0,2.0],[24.0,2.0],[25.0,2.0]],[[0.0,3.0],[1.0,3.0],[2.0,3.0],[3.0,3.0],[4.0,3.0],[5.0,3.0],[6.0,3.0],[7.0,3.0],[8.0,3.0],[9.0,3.0],[10.0,3.0],[11.0,3.0],[12.0,3.0],[13.0,3.0],[14.0,3.0],[15.0,3.0],[16.0,3.0],[17.0,3.0],[18.0,3.0],[19.0,3.0],[20.0,3.0],[21.0,3.0],[22.0,3.0],[23.0,3.0],[24.0,3.0],[25.0,3.0]],[[0.0,4.0],[1.0,4.0],[2.0,4.0],[3.0,4.0],[4.0,4.0],[5.0,4.0],[6.0,4.0],[7.0,4.0],[8.0,4.0],[9.0,4.0],[10.0,4.0],[11.0,4.0],[12.0,4.0],[13.0,4.0],[14.0,4.0],[15.0,4.0],[16.0,4.0],[17.0,4.0],[18.0,4.0],[19.0,4.0],[20.0,4.0],[21.0,4.0],[22.0,4.0],[23.0,4.0],[24.0,4.0],[25.0,4.0]],[[0.0,5.0],[1.0,5.0],[2.0,5.0],[3.0,5.0],[4.0,5.0],[5.0,5.0],[6.0,5.0],[7.0,5.0],[8.0,5.0],[9.0,5.0],[10.0,5.0],[11.0,5.0],[12.0,5.0],[13.0,5.0],[14.0,5.0],[15.0,5.0],[16.0,5.0],[17.0,5.0],[18.0,5.0],[19.0,5.0],[20.0,5.0],[21.0,5.0],[22.0,5.0],[23.0,5.0],[24.0,5.0],[25.0,5.0]],[[0.0,6.0],[1.0,6.0],[2.0,6.0],[3.0,6.0],[4.0,6.0],[5.0,6.0],[6.0,6.0],[7.0,6.0],[8.0,6.0],[9.0,6.0],[10.0,6.0],[11.0,6.0],[12.0,6.0],[13.0,6.0],[14.0,6.0],[15.0,6.0],[16.0,6.0],[17.0,6.0],[18.0,6.0],[19.0,6.0],[20.0,6.0],[21.0,6.0],[22.0,6.0],[23.0,6.0],[24.0,6.0],[25.0,6.0]],[[0.0,7.0],[1.0,7.0],[2.0,7.0],[3.0,7.0],[4.0,7.0],[5.0,7.0],[6.0,7.0],[7.0,7.0],[8.0,7.0],[9.0,7.0],[10.0,7.0],[11.0,7.0],[12.0,7.0],[13.0,7.0],[14.0,7.0],[15.0,7.0],[16.0,7.0],[17.0,7.0],[18.0,7.0],[19.0,7.0],[20.0,7.0],[21.0,7.0],[22.0,7.0],[23.0,7.0],[24.0,7.0],[25.0,7.0]],[[0.0,8.0],[1.0,8.0],[2.0,8.0],[3.0,8.0],[4.0,8.0],[5.0,8.0],[6.0,8.0],[7.0,8.0],[8.0,8.0],[9.0,8.0],[10.0,8.0],[11.0,8.0],[12.0,8.0],[13.0,8.0],[14.0,8.0],[15.0,8.0],[16.0,8.0],[17.0,8.0],[18.0,8.0],[19.0,8.0],[20.0,8.0],[21.0,8.0],[22.0,8.0],[23.0,8.0],[24.0,8.0],[25.0,8.0]],[[0.0,9.0],[1.0,9.0],[2.0,9.0],[3.0,9.0],[4.0,9.0],[5.0,9.0],[6.0,9.0],[7.0,9.0],[8.0,9.0],[9.0,9.0],[10.0,9.0],[11.0,9.0],[12.0,9.0],[13.0,9.0],[14.0,9.0],[15.0,9.0],[16.0,9.0],[17.0,9.0],[18.0,9.0],[19.0,9.0],[20.0,9.0],[21.0,9.0],[22.0,9.0],[23.0,9.0],[24.0,9.0],[25.0,9.0]],[[0.0,10.0],[1.0,10.0],[2.0,10.0],[3.0,10.0],[4.0,10.0],[5.0,10.0],[6.0,10.0],[7.0,10.0],[8.0,10.0],[9.0,10.0],[10.0,10.0],[11.0,10.0],[12.0,10.0],[13.0,10.0],[14.0,10.0],[15.0,10.0],[16.0,10.0],[17.0,10.0],[18.0,10.0],[19.0,10.0],[20.0,10.0],[21.0,10.0],[22.0,10.0],[23.0,10.0],[24.0,10.0],[25.0,10.0]],[[0.0,11.0],[1.0,11.0],[2.0,11.0],[3.0,11.0],[4.0,11.0],[5.0,11.0],[6.0,11.0],[7.0,11.0],[8.0,11.0],[9.0,11.0],[10.0,11.0],[11.0,11.0],[12.0,11.0],[13.0,11.0],[14.0,11.0],[15.0,11.0],[16.0,11.0],[17.0,11.0],[18.0,11.0],[19.0,11.0],[20.0,11.0],[21.0,11.0],[22.0,11.0],[23.0,11.0],[24.0,11.0],[25.0,11.0]],[[0.0,12.0],[1.0,12.0],[2.0,12.0],[3.0,12.0],[4.0,12.0],[5.0,12.0],[6.0,12.0],[7.0,12.0],[8.0,12.0],[9.0,12.0],[10.0,12.0],[11.0,12.0],[12.0,12.0],[13.0,12.0],[14.0,12.0],[15.0,12.0],[16.0,12.0],[17.0,12.0],[18.0,12.0],[19.0,12.0],[20.0,12.0],[21.0,12.0],[22.0,12.0],[23.0,12.0],[24.0,12.0],[25.0,12.0]],[[0.0,13.0],[1.0,13.0],[2.0,13.0],[3.0,13.0],[4.0,13.0],[5.0,13.0],[6.0,13.0],[7.0,13.0],[8.0,13.0],[9.0,13.0],[10.0,13.0],[11.0,13.0],[12.0,13.0],[13.0,13.0],[14.0,13.0],[15.0,13.0],[16.0,13.0],[17.0,13.0],[18.0,13.0],[19.0,13.0],[20.0,13.0],[21.0,13.0],[22.0,13.0],[23.0,13.0],[24.0,13.0],[25.0,13.0]],[[0.0,14.0],[1.0,14.0],[2.0,14.0],[3.0,14.0],[4.0,14.0],[5.0,14.0],[6.0,14.0],[7.0,14.0],[8.0,14.0],[9.0,14.0],[10.0,14.0],[11.0,14.0],[12.0,14.0],[13.0,14.0],[14.0,14.0],[15.0,14.0],[16.0,14.0],[17.0,14.0],[18.0,14.0],[19.0,14.0],[20.0,14.0],[21.0,14.0],[22.0,14.0],[23.0,14.0],[24.0,14.0],[25.0,14.0]],[[0.0,15.0],[1.0,15.0],[2.0,15.0],[3.0,15.0],[4.0,15.0],[5.0,15.0],[6.0,15.0],[7.0,15.0],[8.0,15.0],[9.0,15.0],[10.0,15.0],[11.0,15.0],[12.0,15.0],[13.0,15.0],[14.0,15.0],[15.0,15.0],[16.0,15.0],[17.0,15.0],[18.0,15.0],[19.0,15.0],[20.0,15.0],[21.0,15.0],[22.0,15.0],[23.0,15.0],[24.0,15.0],[25.0,15.0]],[[0.0,16.0],[1.0,16.0],[2.0,16.0],[3.0,16.0],[4.0,16.0],[5.0,16.0],[6.0,16.0],[7.0,16.0],[8.0,16.0],[9.0,16.0],[10.0,16.0],[11.0,16.0],[12.0,16.0],[13.0,16.0],[14.0,16.0],[15.0,16.0],[16.0,16.0],[17.0,16.0],[18.0,16.0],[19.0,16.0],[20.0,16.0],[21.0,16.0],[22.0,16.0],[23.0,16.0],[24.0,16.0],[25.0,16.0]],[[0.0,17.0],[1.0,17.0],[2.0,17.0],[3.0,17.0],[4.0,17.0],[5.0,17.0],[6.0,17.0],[7.0,17.0],[8.0,17.0],[9.0,17.0],[10.0,17.0],[11.0,17.0],[12.0,17.0],[13.0,17.0],[14.0,17.0],[15.0,17.0],[16.0,17.0],[17.0,17.0],[18.0,17.0],[19.0,17.0],[20.0,17.0],[21.0,17.0],[22.0,17.0],[23.0,17.0],[24.0,17.0],[25.0,17.0]],[[0.0,18.0],[1.0,18.0],[2.0,18.0],[3.0,18.0],[4.0,18.0],[5.0,18.0],[6.0,18.0],[7.0,18.0],[8.0,18.0],[9.0,18.0],[10.0,18.0],[11.0,18.0],[12.0,18.0],[13.0,18.0],[14.0,18.0],[15.0,18.0],[16.0,18.0],[17.0,18.0],[18.0,18.0],[19.0,18.0],[20.0,18.0],[21.0,18.0],[22.0,18.0],[23.0,18.0],[24.0,18.0],[25.0,18.0]],[[0.0,19.0],[1.0,19.0],[2.0,19.0],[3.0,19.0],[4.0,19.0],[5.0,19.0],[6.0,19.0],[7.0,19.0],[8.0,19.0],[9.0,19.0],[10.0,19.0],[11.0,19.0],[12.0,19.0],[13.0,19.0],[14.0,19.0],[15.0,19.0],[16.0,19.0],[17.0,19.0],[18.0,19.0],[19.0,19.0],[20.0,19.0],[21.0,19.0],[22.0,19.0],[23.0,19.0],[24.0,19.0],[25.0,19.0]],[[0.0,20.0],[1.0,20.0],[2.0,20.0],[3.0,20.0],[4.0,20.0],[5.0,20.0],[6.0,20.0],[7.0,20.0],[8.0,20.0],[9.0,20.0],[10.0,20.0],[11.0,20.0],[12.0,20.0],[13.0,20.0],[14.0,20.0],[15.0,20.0],[16.0,20.0],[17.0,20.0],[18.0,20.0],[19.0,20.0],[20.0,20.0],[21.0,20.0],[22.0,20.0],[23.0,20.0],[24.0,20.0],[25.0,20.0]],[[0.0,21.0],[1.0,21.0],[2.0,21.0],[3.0,21.0],[4.0,21.0],[5.0,21.0],[6.0,21.0],[7.0,21.0],[8.0,21.0],[9.0,21.0],[10.0,21.0],[11.0,21.0],[12.0,21.0],[13.0,21.0],[14.0,21.0],[15.0,21.0],[16.0,21.0],[17.0,21.0],[18.0,21.0],[19.0,21.0],[20.0,21.0],[21.0,21.0],[22.0,21.0],[23.0,21.0],[24.0,21.0],[25.0,21.0]],[[0.0,22.0],[1.0,22.0],[2.0,22.0],[3.0,22.0],[4.0,22.0],[5.0,22.0],[6.0,22.0],[7.0,22.0],[8.0,22.0],[9.0,22.0],[10.0,22.0],[11.0,22.0],[12.0,22.0],[13.0,22.0],[14.0,22.0],[15.0,22.0],[16.0,22.0],[17.0,22.0],[18.0,22.0],[19.0,22.0],[20.0,22.0],[21.0,22.0],[22.0,22.0],[23.0,22.0],[24.0,22.0],[25.0,22.0]],[[0.0,23.0],[1.0,23.0],[2.0,23.0],[3.0,23.0],[4.0,23.0],[5.0,23.0],[6.0,23.0],[7.0,23.0],[8.0,23.0],[9.0,23.0],[10.0,23.0],[11.0,23.0],[12.0,23.0],[13.0,23.0],[14.0,23.0],[15.0,23.0],[16.0,23.0],[17.0,23.0],[18.0,23.0],[19.0,23.0],[20.0,23.0],[21.0,23.0],[22.0,23.0],[23.0,23.0],[24.0,23.0],[25.0,23.0]],[[0.0,24.0],[1.0,24.0],[2.0,24.0],[3.0,24.0],[4.0,24.0],[5.0,24.0],[6.0,24.0],[7.0,24.0],[8.0,24.0],[9.0,24.0],[10.0,24.0],[11.0,24.0],[12.0,24.0],[13.0,24.0],[14.0,24.0],[15.0,24.0],[16.0,24.0],[17.0,24.0],[18.0,24.0],[19.0,24.0],[20.0,24.0],[21.0,24.0],[22.0,24.0],[23.0,24.0],[24.0,24.0],[25.0,24.0]],[[0.0,25.0],[1.0,25.0],[2.0,25.0],[3.0,25.0],[4.0,25.0],[5.0,25.0],[6.0,25.0],[7.0,25.0],[8.0,25.0],[9.0,25.0],[10.0,25.0],[11.0,25.0],[12.0,25.0],[13.0,25.0],[14.0,25.0],[15.0,25.0],[16.0,25.0],[17.0,25.0],[18.0,25.0],[19.0,25.0],[20.0,25.0],[21.0,25.0],[22.0,25.0],[23.0,25.0],[24.0,25.0],[25.0,25.0]]]]], dtype=torch.float, device='cpu')
        output_module_8 = self.module_8(data=1, dtype=torch.int64, device='cpu')
        output_module_9 = self.module_9(data=1, dtype=torch.int64, device='cpu')
        output_module_10 = self.module_10(data=[[[[[0.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0],[4.0,0.0],[5.0,0.0],[6.0,0.0],[7.0,0.0],[8.0,0.0],[9.0,0.0],[10.0,0.0],[11.0,0.0],[12.0,0.0],[13.0,0.0],[14.0,0.0],[15.0,0.0],[16.0,0.0],[17.0,0.0],[18.0,0.0],[19.0,0.0],[20.0,0.0],[21.0,0.0],[22.0,0.0],[23.0,0.0],[24.0,0.0],[25.0,0.0],[26.0,0.0],[27.0,0.0],[28.0,0.0],[29.0,0.0],[30.0,0.0],[31.0,0.0],[32.0,0.0],[33.0,0.0],[34.0,0.0],[35.0,0.0],[36.0,0.0],[37.0,0.0],[38.0,0.0],[39.0,0.0],[40.0,0.0],[41.0,0.0],[42.0,0.0],[43.0,0.0],[44.0,0.0],[45.0,0.0],[46.0,0.0],[47.0,0.0],[48.0,0.0],[49.0,0.0],[50.0,0.0],[51.0,0.0]],[[0.0,1.0],[1.0,1.0],[2.0,1.0],[3.0,1.0],[4.0,1.0],[5.0,1.0],[6.0,1.0],[7.0,1.0],[8.0,1.0],[9.0,1.0],[10.0,1.0],[11.0,1.0],[12.0,1.0],[13.0,1.0],[14.0,1.0],[15.0,1.0],[16.0,1.0],[17.0,1.0],[18.0,1.0],[19.0,1.0],[20.0,1.0],[21.0,1.0],[22.0,1.0],[23.0,1.0],[24.0,1.0],[25.0,1.0],[26.0,1.0],[27.0,1.0],[28.0,1.0],[29.0,1.0],[30.0,1.0],[31.0,1.0],[32.0,1.0],[33.0,1.0],[34.0,1.0],[35.0,1.0],[36.0,1.0],[37.0,1.0],[38.0,1.0],[39.0,1.0],[40.0,1.0],[41.0,1.0],[42.0,1.0],[43.0,1.0],[44.0,1.0],[45.0,1.0],[46.0,1.0],[47.0,1.0],[48.0,1.0],[49.0,1.0],[50.0,1.0],[51.0,1.0]],[[0.0,2.0],[1.0,2.0],[2.0,2.0],[3.0,2.0],[4.0,2.0],[5.0,2.0],[6.0,2.0],[7.0,2.0],[8.0,2.0],[9.0,2.0],[10.0,2.0],[11.0,2.0],[12.0,2.0],[13.0,2.0],[14.0,2.0],[15.0,2.0],[16.0,2.0],[17.0,2.0],[18.0,2.0],[19.0,2.0],[20.0,2.0],[21.0,2.0],[22.0,2.0],[23.0,2.0],[24.0,2.0],[25.0,2.0],[26.0,2.0],[27.0,2.0],[28.0,2.0],[29.0,2.0],[30.0,2.0],[31.0,2.0],[32.0,2.0],[33.0,2.0],[34.0,2.0],[35.0,2.0],[36.0,2.0],[37.0,2.0],[38.0,2.0],[39.0,2.0],[40.0,2.0],[41.0,2.0],[42.0,2.0],[43.0,2.0],[44.0,2.0],[45.0,2.0],[46.0,2.0],[47.0,2.0],[48.0,2.0],[49.0,2.0],[50.0,2.0],[51.0,2.0]],[[0.0,3.0],[1.0,3.0],[2.0,3.0],[3.0,3.0],[4.0,3.0],[5.0,3.0],[6.0,3.0],[7.0,3.0],[8.0,3.0],[9.0,3.0],[10.0,3.0],[11.0,3.0],[12.0,3.0],[13.0,3.0],[14.0,3.0],[15.0,3.0],[16.0,3.0],[17.0,3.0],[18.0,3.0],[19.0,3.0],[20.0,3.0],[21.0,3.0],[22.0,3.0],[23.0,3.0],[24.0,3.0],[25.0,3.0],[26.0,3.0],[27.0,3.0],[28.0,3.0],[29.0,3.0],[30.0,3.0],[31.0,3.0],[32.0,3.0],[33.0,3.0],[34.0,3.0],[35.0,3.0],[36.0,3.0],[37.0,3.0],[38.0,3.0],[39.0,3.0],[40.0,3.0],[41.0,3.0],[42.0,3.0],[43.0,3.0],[44.0,3.0],[45.0,3.0],[46.0,3.0],[47.0,3.0],[48.0,3.0],[49.0,3.0],[50.0,3.0],[51.0,3.0]],[[0.0,4.0],[1.0,4.0],[2.0,4.0],[3.0,4.0],[4.0,4.0],[5.0,4.0],[6.0,4.0],[7.0,4.0],[8.0,4.0],[9.0,4.0],[10.0,4.0],[11.0,4.0],[12.0,4.0],[13.0,4.0],[14.0,4.0],[15.0,4.0],[16.0,4.0],[17.0,4.0],[18.0,4.0],[19.0,4.0],[20.0,4.0],[21.0,4.0],[22.0,4.0],[23.0,4.0],[24.0,4.0],[25.0,4.0],[26.0,4.0],[27.0,4.0],[28.0,4.0],[29.0,4.0],[30.0,4.0],[31.0,4.0],[32.0,4.0],[33.0,4.0],[34.0,4.0],[35.0,4.0],[36.0,4.0],[37.0,4.0],[38.0,4.0],[39.0,4.0],[40.0,4.0],[41.0,4.0],[42.0,4.0],[43.0,4.0],[44.0,4.0],[45.0,4.0],[46.0,4.0],[47.0,4.0],[48.0,4.0],[49.0,4.0],[50.0,4.0],[51.0,4.0]],[[0.0,5.0],[1.0,5.0],[2.0,5.0],[3.0,5.0],[4.0,5.0],[5.0,5.0],[6.0,5.0],[7.0,5.0],[8.0,5.0],[9.0,5.0],[10.0,5.0],[11.0,5.0],[12.0,5.0],[13.0,5.0],[14.0,5.0],[15.0,5.0],[16.0,5.0],[17.0,5.0],[18.0,5.0],[19.0,5.0],[20.0,5.0],[21.0,5.0],[22.0,5.0],[23.0,5.0],[24.0,5.0],[25.0,5.0],[26.0,5.0],[27.0,5.0],[28.0,5.0],[29.0,5.0],[30.0,5.0],[31.0,5.0],[32.0,5.0],[33.0,5.0],[34.0,5.0],[35.0,5.0],[36.0,5.0],[37.0,5.0],[38.0,5.0],[39.0,5.0],[40.0,5.0],[41.0,5.0],[42.0,5.0],[43.0,5.0],[44.0,5.0],[45.0,5.0],[46.0,5.0],[47.0,5.0],[48.0,5.0],[49.0,5.0],[50.0,5.0],[51.0,5.0]],[[0.0,6.0],[1.0,6.0],[2.0,6.0],[3.0,6.0],[4.0,6.0],[5.0,6.0],[6.0,6.0],[7.0,6.0],[8.0,6.0],[9.0,6.0],[10.0,6.0],[11.0,6.0],[12.0,6.0],[13.0,6.0],[14.0,6.0],[15.0,6.0],[16.0,6.0],[17.0,6.0],[18.0,6.0],[19.0,6.0],[20.0,6.0],[21.0,6.0],[22.0,6.0],[23.0,6.0],[24.0,6.0],[25.0,6.0],[26.0,6.0],[27.0,6.0],[28.0,6.0],[29.0,6.0],[30.0,6.0],[31.0,6.0],[32.0,6.0],[33.0,6.0],[34.0,6.0],[35.0,6.0],[36.0,6.0],[37.0,6.0],[38.0,6.0],[39.0,6.0],[40.0,6.0],[41.0,6.0],[42.0,6.0],[43.0,6.0],[44.0,6.0],[45.0,6.0],[46.0,6.0],[47.0,6.0],[48.0,6.0],[49.0,6.0],[50.0,6.0],[51.0,6.0]],[[0.0,7.0],[1.0,7.0],[2.0,7.0],[3.0,7.0],[4.0,7.0],[5.0,7.0],[6.0,7.0],[7.0,7.0],[8.0,7.0],[9.0,7.0],[10.0,7.0],[11.0,7.0],[12.0,7.0],[13.0,7.0],[14.0,7.0],[15.0,7.0],[16.0,7.0],[17.0,7.0],[18.0,7.0],[19.0,7.0],[20.0,7.0],[21.0,7.0],[22.0,7.0],[23.0,7.0],[24.0,7.0],[25.0,7.0],[26.0,7.0],[27.0,7.0],[28.0,7.0],[29.0,7.0],[30.0,7.0],[31.0,7.0],[32.0,7.0],[33.0,7.0],[34.0,7.0],[35.0,7.0],[36.0,7.0],[37.0,7.0],[38.0,7.0],[39.0,7.0],[40.0,7.0],[41.0,7.0],[42.0,7.0],[43.0,7.0],[44.0,7.0],[45.0,7.0],[46.0,7.0],[47.0,7.0],[48.0,7.0],[49.0,7.0],[50.0,7.0],[51.0,7.0]],[[0.0,8.0],[1.0,8.0],[2.0,8.0],[3.0,8.0],[4.0,8.0],[5.0,8.0],[6.0,8.0],[7.0,8.0],[8.0,8.0],[9.0,8.0],[10.0,8.0],[11.0,8.0],[12.0,8.0],[13.0,8.0],[14.0,8.0],[15.0,8.0],[16.0,8.0],[17.0,8.0],[18.0,8.0],[19.0,8.0],[20.0,8.0],[21.0,8.0],[22.0,8.0],[23.0,8.0],[24.0,8.0],[25.0,8.0],[26.0,8.0],[27.0,8.0],[28.0,8.0],[29.0,8.0],[30.0,8.0],[31.0,8.0],[32.0,8.0],[33.0,8.0],[34.0,8.0],[35.0,8.0],[36.0,8.0],[37.0,8.0],[38.0,8.0],[39.0,8.0],[40.0,8.0],[41.0,8.0],[42.0,8.0],[43.0,8.0],[44.0,8.0],[45.0,8.0],[46.0,8.0],[47.0,8.0],[48.0,8.0],[49.0,8.0],[50.0,8.0],[51.0,8.0]],[[0.0,9.0],[1.0,9.0],[2.0,9.0],[3.0,9.0],[4.0,9.0],[5.0,9.0],[6.0,9.0],[7.0,9.0],[8.0,9.0],[9.0,9.0],[10.0,9.0],[11.0,9.0],[12.0,9.0],[13.0,9.0],[14.0,9.0],[15.0,9.0],[16.0,9.0],[17.0,9.0],[18.0,9.0],[19.0,9.0],[20.0,9.0],[21.0,9.0],[22.0,9.0],[23.0,9.0],[24.0,9.0],[25.0,9.0],[26.0,9.0],[27.0,9.0],[28.0,9.0],[29.0,9.0],[30.0,9.0],[31.0,9.0],[32.0,9.0],[33.0,9.0],[34.0,9.0],[35.0,9.0],[36.0,9.0],[37.0,9.0],[38.0,9.0],[39.0,9.0],[40.0,9.0],[41.0,9.0],[42.0,9.0],[43.0,9.0],[44.0,9.0],[45.0,9.0],[46.0,9.0],[47.0,9.0],[48.0,9.0],[49.0,9.0],[50.0,9.0],[51.0,9.0]],[[0.0,10.0],[1.0,10.0],[2.0,10.0],[3.0,10.0],[4.0,10.0],[5.0,10.0],[6.0,10.0],[7.0,10.0],[8.0,10.0],[9.0,10.0],[10.0,10.0],[11.0,10.0],[12.0,10.0],[13.0,10.0],[14.0,10.0],[15.0,10.0],[16.0,10.0],[17.0,10.0],[18.0,10.0],[19.0,10.0],[20.0,10.0],[21.0,10.0],[22.0,10.0],[23.0,10.0],[24.0,10.0],[25.0,10.0],[26.0,10.0],[27.0,10.0],[28.0,10.0],[29.0,10.0],[30.0,10.0],[31.0,10.0],[32.0,10.0],[33.0,10.0],[34.0,10.0],[35.0,10.0],[36.0,10.0],[37.0,10.0],[38.0,10.0],[39.0,10.0],[40.0,10.0],[41.0,10.0],[42.0,10.0],[43.0,10.0],[44.0,10.0],[45.0,10.0],[46.0,10.0],[47.0,10.0],[48.0,10.0],[49.0,10.0],[50.0,10.0],[51.0,10.0]],[[0.0,11.0],[1.0,11.0],[2.0,11.0],[3.0,11.0],[4.0,11.0],[5.0,11.0],[6.0,11.0],[7.0,11.0],[8.0,11.0],[9.0,11.0],[10.0,11.0],[11.0,11.0],[12.0,11.0],[13.0,11.0],[14.0,11.0],[15.0,11.0],[16.0,11.0],[17.0,11.0],[18.0,11.0],[19.0,11.0],[20.0,11.0],[21.0,11.0],[22.0,11.0],[23.0,11.0],[24.0,11.0],[25.0,11.0],[26.0,11.0],[27.0,11.0],[28.0,11.0],[29.0,11.0],[30.0,11.0],[31.0,11.0],[32.0,11.0],[33.0,11.0],[34.0,11.0],[35.0,11.0],[36.0,11.0],[37.0,11.0],[38.0,11.0],[39.0,11.0],[40.0,11.0],[41.0,11.0],[42.0,11.0],[43.0,11.0],[44.0,11.0],[45.0,11.0],[46.0,11.0],[47.0,11.0],[48.0,11.0],[49.0,11.0],[50.0,11.0],[51.0,11.0]],[[0.0,12.0],[1.0,12.0],[2.0,12.0],[3.0,12.0],[4.0,12.0],[5.0,12.0],[6.0,12.0],[7.0,12.0],[8.0,12.0],[9.0,12.0],[10.0,12.0],[11.0,12.0],[12.0,12.0],[13.0,12.0],[14.0,12.0],[15.0,12.0],[16.0,12.0],[17.0,12.0],[18.0,12.0],[19.0,12.0],[20.0,12.0],[21.0,12.0],[22.0,12.0],[23.0,12.0],[24.0,12.0],[25.0,12.0],[26.0,12.0],[27.0,12.0],[28.0,12.0],[29.0,12.0],[30.0,12.0],[31.0,12.0],[32.0,12.0],[33.0,12.0],[34.0,12.0],[35.0,12.0],[36.0,12.0],[37.0,12.0],[38.0,12.0],[39.0,12.0],[40.0,12.0],[41.0,12.0],[42.0,12.0],[43.0,12.0],[44.0,12.0],[45.0,12.0],[46.0,12.0],[47.0,12.0],[48.0,12.0],[49.0,12.0],[50.0,12.0],[51.0,12.0]],[[0.0,13.0],[1.0,13.0],[2.0,13.0],[3.0,13.0],[4.0,13.0],[5.0,13.0],[6.0,13.0],[7.0,13.0],[8.0,13.0],[9.0,13.0],[10.0,13.0],[11.0,13.0],[12.0,13.0],[13.0,13.0],[14.0,13.0],[15.0,13.0],[16.0,13.0],[17.0,13.0],[18.0,13.0],[19.0,13.0],[20.0,13.0],[21.0,13.0],[22.0,13.0],[23.0,13.0],[24.0,13.0],[25.0,13.0],[26.0,13.0],[27.0,13.0],[28.0,13.0],[29.0,13.0],[30.0,13.0],[31.0,13.0],[32.0,13.0],[33.0,13.0],[34.0,13.0],[35.0,13.0],[36.0,13.0],[37.0,13.0],[38.0,13.0],[39.0,13.0],[40.0,13.0],[41.0,13.0],[42.0,13.0],[43.0,13.0],[44.0,13.0],[45.0,13.0],[46.0,13.0],[47.0,13.0],[48.0,13.0],[49.0,13.0],[50.0,13.0],[51.0,13.0]],[[0.0,14.0],[1.0,14.0],[2.0,14.0],[3.0,14.0],[4.0,14.0],[5.0,14.0],[6.0,14.0],[7.0,14.0],[8.0,14.0],[9.0,14.0],[10.0,14.0],[11.0,14.0],[12.0,14.0],[13.0,14.0],[14.0,14.0],[15.0,14.0],[16.0,14.0],[17.0,14.0],[18.0,14.0],[19.0,14.0],[20.0,14.0],[21.0,14.0],[22.0,14.0],[23.0,14.0],[24.0,14.0],[25.0,14.0],[26.0,14.0],[27.0,14.0],[28.0,14.0],[29.0,14.0],[30.0,14.0],[31.0,14.0],[32.0,14.0],[33.0,14.0],[34.0,14.0],[35.0,14.0],[36.0,14.0],[37.0,14.0],[38.0,14.0],[39.0,14.0],[40.0,14.0],[41.0,14.0],[42.0,14.0],[43.0,14.0],[44.0,14.0],[45.0,14.0],[46.0,14.0],[47.0,14.0],[48.0,14.0],[49.0,14.0],[50.0,14.0],[51.0,14.0]],[[0.0,15.0],[1.0,15.0],[2.0,15.0],[3.0,15.0],[4.0,15.0],[5.0,15.0],[6.0,15.0],[7.0,15.0],[8.0,15.0],[9.0,15.0],[10.0,15.0],[11.0,15.0],[12.0,15.0],[13.0,15.0],[14.0,15.0],[15.0,15.0],[16.0,15.0],[17.0,15.0],[18.0,15.0],[19.0,15.0],[20.0,15.0],[21.0,15.0],[22.0,15.0],[23.0,15.0],[24.0,15.0],[25.0,15.0],[26.0,15.0],[27.0,15.0],[28.0,15.0],[29.0,15.0],[30.0,15.0],[31.0,15.0],[32.0,15.0],[33.0,15.0],[34.0,15.0],[35.0,15.0],[36.0,15.0],[37.0,15.0],[38.0,15.0],[39.0,15.0],[40.0,15.0],[41.0,15.0],[42.0,15.0],[43.0,15.0],[44.0,15.0],[45.0,15.0],[46.0,15.0],[47.0,15.0],[48.0,15.0],[49.0,15.0],[50.0,15.0],[51.0,15.0]],[[0.0,16.0],[1.0,16.0],[2.0,16.0],[3.0,16.0],[4.0,16.0],[5.0,16.0],[6.0,16.0],[7.0,16.0],[8.0,16.0],[9.0,16.0],[10.0,16.0],[11.0,16.0],[12.0,16.0],[13.0,16.0],[14.0,16.0],[15.0,16.0],[16.0,16.0],[17.0,16.0],[18.0,16.0],[19.0,16.0],[20.0,16.0],[21.0,16.0],[22.0,16.0],[23.0,16.0],[24.0,16.0],[25.0,16.0],[26.0,16.0],[27.0,16.0],[28.0,16.0],[29.0,16.0],[30.0,16.0],[31.0,16.0],[32.0,16.0],[33.0,16.0],[34.0,16.0],[35.0,16.0],[36.0,16.0],[37.0,16.0],[38.0,16.0],[39.0,16.0],[40.0,16.0],[41.0,16.0],[42.0,16.0],[43.0,16.0],[44.0,16.0],[45.0,16.0],[46.0,16.0],[47.0,16.0],[48.0,16.0],[49.0,16.0],[50.0,16.0],[51.0,16.0]],[[0.0,17.0],[1.0,17.0],[2.0,17.0],[3.0,17.0],[4.0,17.0],[5.0,17.0],[6.0,17.0],[7.0,17.0],[8.0,17.0],[9.0,17.0],[10.0,17.0],[11.0,17.0],[12.0,17.0],[13.0,17.0],[14.0,17.0],[15.0,17.0],[16.0,17.0],[17.0,17.0],[18.0,17.0],[19.0,17.0],[20.0,17.0],[21.0,17.0],[22.0,17.0],[23.0,17.0],[24.0,17.0],[25.0,17.0],[26.0,17.0],[27.0,17.0],[28.0,17.0],[29.0,17.0],[30.0,17.0],[31.0,17.0],[32.0,17.0],[33.0,17.0],[34.0,17.0],[35.0,17.0],[36.0,17.0],[37.0,17.0],[38.0,17.0],[39.0,17.0],[40.0,17.0],[41.0,17.0],[42.0,17.0],[43.0,17.0],[44.0,17.0],[45.0,17.0],[46.0,17.0],[47.0,17.0],[48.0,17.0],[49.0,17.0],[50.0,17.0],[51.0,17.0]],[[0.0,18.0],[1.0,18.0],[2.0,18.0],[3.0,18.0],[4.0,18.0],[5.0,18.0],[6.0,18.0],[7.0,18.0],[8.0,18.0],[9.0,18.0],[10.0,18.0],[11.0,18.0],[12.0,18.0],[13.0,18.0],[14.0,18.0],[15.0,18.0],[16.0,18.0],[17.0,18.0],[18.0,18.0],[19.0,18.0],[20.0,18.0],[21.0,18.0],[22.0,18.0],[23.0,18.0],[24.0,18.0],[25.0,18.0],[26.0,18.0],[27.0,18.0],[28.0,18.0],[29.0,18.0],[30.0,18.0],[31.0,18.0],[32.0,18.0],[33.0,18.0],[34.0,18.0],[35.0,18.0],[36.0,18.0],[37.0,18.0],[38.0,18.0],[39.0,18.0],[40.0,18.0],[41.0,18.0],[42.0,18.0],[43.0,18.0],[44.0,18.0],[45.0,18.0],[46.0,18.0],[47.0,18.0],[48.0,18.0],[49.0,18.0],[50.0,18.0],[51.0,18.0]],[[0.0,19.0],[1.0,19.0],[2.0,19.0],[3.0,19.0],[4.0,19.0],[5.0,19.0],[6.0,19.0],[7.0,19.0],[8.0,19.0],[9.0,19.0],[10.0,19.0],[11.0,19.0],[12.0,19.0],[13.0,19.0],[14.0,19.0],[15.0,19.0],[16.0,19.0],[17.0,19.0],[18.0,19.0],[19.0,19.0],[20.0,19.0],[21.0,19.0],[22.0,19.0],[23.0,19.0],[24.0,19.0],[25.0,19.0],[26.0,19.0],[27.0,19.0],[28.0,19.0],[29.0,19.0],[30.0,19.0],[31.0,19.0],[32.0,19.0],[33.0,19.0],[34.0,19.0],[35.0,19.0],[36.0,19.0],[37.0,19.0],[38.0,19.0],[39.0,19.0],[40.0,19.0],[41.0,19.0],[42.0,19.0],[43.0,19.0],[44.0,19.0],[45.0,19.0],[46.0,19.0],[47.0,19.0],[48.0,19.0],[49.0,19.0],[50.0,19.0],[51.0,19.0]],[[0.0,20.0],[1.0,20.0],[2.0,20.0],[3.0,20.0],[4.0,20.0],[5.0,20.0],[6.0,20.0],[7.0,20.0],[8.0,20.0],[9.0,20.0],[10.0,20.0],[11.0,20.0],[12.0,20.0],[13.0,20.0],[14.0,20.0],[15.0,20.0],[16.0,20.0],[17.0,20.0],[18.0,20.0],[19.0,20.0],[20.0,20.0],[21.0,20.0],[22.0,20.0],[23.0,20.0],[24.0,20.0],[25.0,20.0],[26.0,20.0],[27.0,20.0],[28.0,20.0],[29.0,20.0],[30.0,20.0],[31.0,20.0],[32.0,20.0],[33.0,20.0],[34.0,20.0],[35.0,20.0],[36.0,20.0],[37.0,20.0],[38.0,20.0],[39.0,20.0],[40.0,20.0],[41.0,20.0],[42.0,20.0],[43.0,20.0],[44.0,20.0],[45.0,20.0],[46.0,20.0],[47.0,20.0],[48.0,20.0],[49.0,20.0],[50.0,20.0],[51.0,20.0]],[[0.0,21.0],[1.0,21.0],[2.0,21.0],[3.0,21.0],[4.0,21.0],[5.0,21.0],[6.0,21.0],[7.0,21.0],[8.0,21.0],[9.0,21.0],[10.0,21.0],[11.0,21.0],[12.0,21.0],[13.0,21.0],[14.0,21.0],[15.0,21.0],[16.0,21.0],[17.0,21.0],[18.0,21.0],[19.0,21.0],[20.0,21.0],[21.0,21.0],[22.0,21.0],[23.0,21.0],[24.0,21.0],[25.0,21.0],[26.0,21.0],[27.0,21.0],[28.0,21.0],[29.0,21.0],[30.0,21.0],[31.0,21.0],[32.0,21.0],[33.0,21.0],[34.0,21.0],[35.0,21.0],[36.0,21.0],[37.0,21.0],[38.0,21.0],[39.0,21.0],[40.0,21.0],[41.0,21.0],[42.0,21.0],[43.0,21.0],[44.0,21.0],[45.0,21.0],[46.0,21.0],[47.0,21.0],[48.0,21.0],[49.0,21.0],[50.0,21.0],[51.0,21.0]],[[0.0,22.0],[1.0,22.0],[2.0,22.0],[3.0,22.0],[4.0,22.0],[5.0,22.0],[6.0,22.0],[7.0,22.0],[8.0,22.0],[9.0,22.0],[10.0,22.0],[11.0,22.0],[12.0,22.0],[13.0,22.0],[14.0,22.0],[15.0,22.0],[16.0,22.0],[17.0,22.0],[18.0,22.0],[19.0,22.0],[20.0,22.0],[21.0,22.0],[22.0,22.0],[23.0,22.0],[24.0,22.0],[25.0,22.0],[26.0,22.0],[27.0,22.0],[28.0,22.0],[29.0,22.0],[30.0,22.0],[31.0,22.0],[32.0,22.0],[33.0,22.0],[34.0,22.0],[35.0,22.0],[36.0,22.0],[37.0,22.0],[38.0,22.0],[39.0,22.0],[40.0,22.0],[41.0,22.0],[42.0,22.0],[43.0,22.0],[44.0,22.0],[45.0,22.0],[46.0,22.0],[47.0,22.0],[48.0,22.0],[49.0,22.0],[50.0,22.0],[51.0,22.0]],[[0.0,23.0],[1.0,23.0],[2.0,23.0],[3.0,23.0],[4.0,23.0],[5.0,23.0],[6.0,23.0],[7.0,23.0],[8.0,23.0],[9.0,23.0],[10.0,23.0],[11.0,23.0],[12.0,23.0],[13.0,23.0],[14.0,23.0],[15.0,23.0],[16.0,23.0],[17.0,23.0],[18.0,23.0],[19.0,23.0],[20.0,23.0],[21.0,23.0],[22.0,23.0],[23.0,23.0],[24.0,23.0],[25.0,23.0],[26.0,23.0],[27.0,23.0],[28.0,23.0],[29.0,23.0],[30.0,23.0],[31.0,23.0],[32.0,23.0],[33.0,23.0],[34.0,23.0],[35.0,23.0],[36.0,23.0],[37.0,23.0],[38.0,23.0],[39.0,23.0],[40.0,23.0],[41.0,23.0],[42.0,23.0],[43.0,23.0],[44.0,23.0],[45.0,23.0],[46.0,23.0],[47.0,23.0],[48.0,23.0],[49.0,23.0],[50.0,23.0],[51.0,23.0]],[[0.0,24.0],[1.0,24.0],[2.0,24.0],[3.0,24.0],[4.0,24.0],[5.0,24.0],[6.0,24.0],[7.0,24.0],[8.0,24.0],[9.0,24.0],[10.0,24.0],[11.0,24.0],[12.0,24.0],[13.0,24.0],[14.0,24.0],[15.0,24.0],[16.0,24.0],[17.0,24.0],[18.0,24.0],[19.0,24.0],[20.0,24.0],[21.0,24.0],[22.0,24.0],[23.0,24.0],[24.0,24.0],[25.0,24.0],[26.0,24.0],[27.0,24.0],[28.0,24.0],[29.0,24.0],[30.0,24.0],[31.0,24.0],[32.0,24.0],[33.0,24.0],[34.0,24.0],[35.0,24.0],[36.0,24.0],[37.0,24.0],[38.0,24.0],[39.0,24.0],[40.0,24.0],[41.0,24.0],[42.0,24.0],[43.0,24.0],[44.0,24.0],[45.0,24.0],[46.0,24.0],[47.0,24.0],[48.0,24.0],[49.0,24.0],[50.0,24.0],[51.0,24.0]],[[0.0,25.0],[1.0,25.0],[2.0,25.0],[3.0,25.0],[4.0,25.0],[5.0,25.0],[6.0,25.0],[7.0,25.0],[8.0,25.0],[9.0,25.0],[10.0,25.0],[11.0,25.0],[12.0,25.0],[13.0,25.0],[14.0,25.0],[15.0,25.0],[16.0,25.0],[17.0,25.0],[18.0,25.0],[19.0,25.0],[20.0,25.0],[21.0,25.0],[22.0,25.0],[23.0,25.0],[24.0,25.0],[25.0,25.0],[26.0,25.0],[27.0,25.0],[28.0,25.0],[29.0,25.0],[30.0,25.0],[31.0,25.0],[32.0,25.0],[33.0,25.0],[34.0,25.0],[35.0,25.0],[36.0,25.0],[37.0,25.0],[38.0,25.0],[39.0,25.0],[40.0,25.0],[41.0,25.0],[42.0,25.0],[43.0,25.0],[44.0,25.0],[45.0,25.0],[46.0,25.0],[47.0,25.0],[48.0,25.0],[49.0,25.0],[50.0,25.0],[51.0,25.0]],[[0.0,26.0],[1.0,26.0],[2.0,26.0],[3.0,26.0],[4.0,26.0],[5.0,26.0],[6.0,26.0],[7.0,26.0],[8.0,26.0],[9.0,26.0],[10.0,26.0],[11.0,26.0],[12.0,26.0],[13.0,26.0],[14.0,26.0],[15.0,26.0],[16.0,26.0],[17.0,26.0],[18.0,26.0],[19.0,26.0],[20.0,26.0],[21.0,26.0],[22.0,26.0],[23.0,26.0],[24.0,26.0],[25.0,26.0],[26.0,26.0],[27.0,26.0],[28.0,26.0],[29.0,26.0],[30.0,26.0],[31.0,26.0],[32.0,26.0],[33.0,26.0],[34.0,26.0],[35.0,26.0],[36.0,26.0],[37.0,26.0],[38.0,26.0],[39.0,26.0],[40.0,26.0],[41.0,26.0],[42.0,26.0],[43.0,26.0],[44.0,26.0],[45.0,26.0],[46.0,26.0],[47.0,26.0],[48.0,26.0],[49.0,26.0],[50.0,26.0],[51.0,26.0]],[[0.0,27.0],[1.0,27.0],[2.0,27.0],[3.0,27.0],[4.0,27.0],[5.0,27.0],[6.0,27.0],[7.0,27.0],[8.0,27.0],[9.0,27.0],[10.0,27.0],[11.0,27.0],[12.0,27.0],[13.0,27.0],[14.0,27.0],[15.0,27.0],[16.0,27.0],[17.0,27.0],[18.0,27.0],[19.0,27.0],[20.0,27.0],[21.0,27.0],[22.0,27.0],[23.0,27.0],[24.0,27.0],[25.0,27.0],[26.0,27.0],[27.0,27.0],[28.0,27.0],[29.0,27.0],[30.0,27.0],[31.0,27.0],[32.0,27.0],[33.0,27.0],[34.0,27.0],[35.0,27.0],[36.0,27.0],[37.0,27.0],[38.0,27.0],[39.0,27.0],[40.0,27.0],[41.0,27.0],[42.0,27.0],[43.0,27.0],[44.0,27.0],[45.0,27.0],[46.0,27.0],[47.0,27.0],[48.0,27.0],[49.0,27.0],[50.0,27.0],[51.0,27.0]],[[0.0,28.0],[1.0,28.0],[2.0,28.0],[3.0,28.0],[4.0,28.0],[5.0,28.0],[6.0,28.0],[7.0,28.0],[8.0,28.0],[9.0,28.0],[10.0,28.0],[11.0,28.0],[12.0,28.0],[13.0,28.0],[14.0,28.0],[15.0,28.0],[16.0,28.0],[17.0,28.0],[18.0,28.0],[19.0,28.0],[20.0,28.0],[21.0,28.0],[22.0,28.0],[23.0,28.0],[24.0,28.0],[25.0,28.0],[26.0,28.0],[27.0,28.0],[28.0,28.0],[29.0,28.0],[30.0,28.0],[31.0,28.0],[32.0,28.0],[33.0,28.0],[34.0,28.0],[35.0,28.0],[36.0,28.0],[37.0,28.0],[38.0,28.0],[39.0,28.0],[40.0,28.0],[41.0,28.0],[42.0,28.0],[43.0,28.0],[44.0,28.0],[45.0,28.0],[46.0,28.0],[47.0,28.0],[48.0,28.0],[49.0,28.0],[50.0,28.0],[51.0,28.0]],[[0.0,29.0],[1.0,29.0],[2.0,29.0],[3.0,29.0],[4.0,29.0],[5.0,29.0],[6.0,29.0],[7.0,29.0],[8.0,29.0],[9.0,29.0],[10.0,29.0],[11.0,29.0],[12.0,29.0],[13.0,29.0],[14.0,29.0],[15.0,29.0],[16.0,29.0],[17.0,29.0],[18.0,29.0],[19.0,29.0],[20.0,29.0],[21.0,29.0],[22.0,29.0],[23.0,29.0],[24.0,29.0],[25.0,29.0],[26.0,29.0],[27.0,29.0],[28.0,29.0],[29.0,29.0],[30.0,29.0],[31.0,29.0],[32.0,29.0],[33.0,29.0],[34.0,29.0],[35.0,29.0],[36.0,29.0],[37.0,29.0],[38.0,29.0],[39.0,29.0],[40.0,29.0],[41.0,29.0],[42.0,29.0],[43.0,29.0],[44.0,29.0],[45.0,29.0],[46.0,29.0],[47.0,29.0],[48.0,29.0],[49.0,29.0],[50.0,29.0],[51.0,29.0]],[[0.0,30.0],[1.0,30.0],[2.0,30.0],[3.0,30.0],[4.0,30.0],[5.0,30.0],[6.0,30.0],[7.0,30.0],[8.0,30.0],[9.0,30.0],[10.0,30.0],[11.0,30.0],[12.0,30.0],[13.0,30.0],[14.0,30.0],[15.0,30.0],[16.0,30.0],[17.0,30.0],[18.0,30.0],[19.0,30.0],[20.0,30.0],[21.0,30.0],[22.0,30.0],[23.0,30.0],[24.0,30.0],[25.0,30.0],[26.0,30.0],[27.0,30.0],[28.0,30.0],[29.0,30.0],[30.0,30.0],[31.0,30.0],[32.0,30.0],[33.0,30.0],[34.0,30.0],[35.0,30.0],[36.0,30.0],[37.0,30.0],[38.0,30.0],[39.0,30.0],[40.0,30.0],[41.0,30.0],[42.0,30.0],[43.0,30.0],[44.0,30.0],[45.0,30.0],[46.0,30.0],[47.0,30.0],[48.0,30.0],[49.0,30.0],[50.0,30.0],[51.0,30.0]],[[0.0,31.0],[1.0,31.0],[2.0,31.0],[3.0,31.0],[4.0,31.0],[5.0,31.0],[6.0,31.0],[7.0,31.0],[8.0,31.0],[9.0,31.0],[10.0,31.0],[11.0,31.0],[12.0,31.0],[13.0,31.0],[14.0,31.0],[15.0,31.0],[16.0,31.0],[17.0,31.0],[18.0,31.0],[19.0,31.0],[20.0,31.0],[21.0,31.0],[22.0,31.0],[23.0,31.0],[24.0,31.0],[25.0,31.0],[26.0,31.0],[27.0,31.0],[28.0,31.0],[29.0,31.0],[30.0,31.0],[31.0,31.0],[32.0,31.0],[33.0,31.0],[34.0,31.0],[35.0,31.0],[36.0,31.0],[37.0,31.0],[38.0,31.0],[39.0,31.0],[40.0,31.0],[41.0,31.0],[42.0,31.0],[43.0,31.0],[44.0,31.0],[45.0,31.0],[46.0,31.0],[47.0,31.0],[48.0,31.0],[49.0,31.0],[50.0,31.0],[51.0,31.0]],[[0.0,32.0],[1.0,32.0],[2.0,32.0],[3.0,32.0],[4.0,32.0],[5.0,32.0],[6.0,32.0],[7.0,32.0],[8.0,32.0],[9.0,32.0],[10.0,32.0],[11.0,32.0],[12.0,32.0],[13.0,32.0],[14.0,32.0],[15.0,32.0],[16.0,32.0],[17.0,32.0],[18.0,32.0],[19.0,32.0],[20.0,32.0],[21.0,32.0],[22.0,32.0],[23.0,32.0],[24.0,32.0],[25.0,32.0],[26.0,32.0],[27.0,32.0],[28.0,32.0],[29.0,32.0],[30.0,32.0],[31.0,32.0],[32.0,32.0],[33.0,32.0],[34.0,32.0],[35.0,32.0],[36.0,32.0],[37.0,32.0],[38.0,32.0],[39.0,32.0],[40.0,32.0],[41.0,32.0],[42.0,32.0],[43.0,32.0],[44.0,32.0],[45.0,32.0],[46.0,32.0],[47.0,32.0],[48.0,32.0],[49.0,32.0],[50.0,32.0],[51.0,32.0]],[[0.0,33.0],[1.0,33.0],[2.0,33.0],[3.0,33.0],[4.0,33.0],[5.0,33.0],[6.0,33.0],[7.0,33.0],[8.0,33.0],[9.0,33.0],[10.0,33.0],[11.0,33.0],[12.0,33.0],[13.0,33.0],[14.0,33.0],[15.0,33.0],[16.0,33.0],[17.0,33.0],[18.0,33.0],[19.0,33.0],[20.0,33.0],[21.0,33.0],[22.0,33.0],[23.0,33.0],[24.0,33.0],[25.0,33.0],[26.0,33.0],[27.0,33.0],[28.0,33.0],[29.0,33.0],[30.0,33.0],[31.0,33.0],[32.0,33.0],[33.0,33.0],[34.0,33.0],[35.0,33.0],[36.0,33.0],[37.0,33.0],[38.0,33.0],[39.0,33.0],[40.0,33.0],[41.0,33.0],[42.0,33.0],[43.0,33.0],[44.0,33.0],[45.0,33.0],[46.0,33.0],[47.0,33.0],[48.0,33.0],[49.0,33.0],[50.0,33.0],[51.0,33.0]],[[0.0,34.0],[1.0,34.0],[2.0,34.0],[3.0,34.0],[4.0,34.0],[5.0,34.0],[6.0,34.0],[7.0,34.0],[8.0,34.0],[9.0,34.0],[10.0,34.0],[11.0,34.0],[12.0,34.0],[13.0,34.0],[14.0,34.0],[15.0,34.0],[16.0,34.0],[17.0,34.0],[18.0,34.0],[19.0,34.0],[20.0,34.0],[21.0,34.0],[22.0,34.0],[23.0,34.0],[24.0,34.0],[25.0,34.0],[26.0,34.0],[27.0,34.0],[28.0,34.0],[29.0,34.0],[30.0,34.0],[31.0,34.0],[32.0,34.0],[33.0,34.0],[34.0,34.0],[35.0,34.0],[36.0,34.0],[37.0,34.0],[38.0,34.0],[39.0,34.0],[40.0,34.0],[41.0,34.0],[42.0,34.0],[43.0,34.0],[44.0,34.0],[45.0,34.0],[46.0,34.0],[47.0,34.0],[48.0,34.0],[49.0,34.0],[50.0,34.0],[51.0,34.0]],[[0.0,35.0],[1.0,35.0],[2.0,35.0],[3.0,35.0],[4.0,35.0],[5.0,35.0],[6.0,35.0],[7.0,35.0],[8.0,35.0],[9.0,35.0],[10.0,35.0],[11.0,35.0],[12.0,35.0],[13.0,35.0],[14.0,35.0],[15.0,35.0],[16.0,35.0],[17.0,35.0],[18.0,35.0],[19.0,35.0],[20.0,35.0],[21.0,35.0],[22.0,35.0],[23.0,35.0],[24.0,35.0],[25.0,35.0],[26.0,35.0],[27.0,35.0],[28.0,35.0],[29.0,35.0],[30.0,35.0],[31.0,35.0],[32.0,35.0],[33.0,35.0],[34.0,35.0],[35.0,35.0],[36.0,35.0],[37.0,35.0],[38.0,35.0],[39.0,35.0],[40.0,35.0],[41.0,35.0],[42.0,35.0],[43.0,35.0],[44.0,35.0],[45.0,35.0],[46.0,35.0],[47.0,35.0],[48.0,35.0],[49.0,35.0],[50.0,35.0],[51.0,35.0]],[[0.0,36.0],[1.0,36.0],[2.0,36.0],[3.0,36.0],[4.0,36.0],[5.0,36.0],[6.0,36.0],[7.0,36.0],[8.0,36.0],[9.0,36.0],[10.0,36.0],[11.0,36.0],[12.0,36.0],[13.0,36.0],[14.0,36.0],[15.0,36.0],[16.0,36.0],[17.0,36.0],[18.0,36.0],[19.0,36.0],[20.0,36.0],[21.0,36.0],[22.0,36.0],[23.0,36.0],[24.0,36.0],[25.0,36.0],[26.0,36.0],[27.0,36.0],[28.0,36.0],[29.0,36.0],[30.0,36.0],[31.0,36.0],[32.0,36.0],[33.0,36.0],[34.0,36.0],[35.0,36.0],[36.0,36.0],[37.0,36.0],[38.0,36.0],[39.0,36.0],[40.0,36.0],[41.0,36.0],[42.0,36.0],[43.0,36.0],[44.0,36.0],[45.0,36.0],[46.0,36.0],[47.0,36.0],[48.0,36.0],[49.0,36.0],[50.0,36.0],[51.0,36.0]],[[0.0,37.0],[1.0,37.0],[2.0,37.0],[3.0,37.0],[4.0,37.0],[5.0,37.0],[6.0,37.0],[7.0,37.0],[8.0,37.0],[9.0,37.0],[10.0,37.0],[11.0,37.0],[12.0,37.0],[13.0,37.0],[14.0,37.0],[15.0,37.0],[16.0,37.0],[17.0,37.0],[18.0,37.0],[19.0,37.0],[20.0,37.0],[21.0,37.0],[22.0,37.0],[23.0,37.0],[24.0,37.0],[25.0,37.0],[26.0,37.0],[27.0,37.0],[28.0,37.0],[29.0,37.0],[30.0,37.0],[31.0,37.0],[32.0,37.0],[33.0,37.0],[34.0,37.0],[35.0,37.0],[36.0,37.0],[37.0,37.0],[38.0,37.0],[39.0,37.0],[40.0,37.0],[41.0,37.0],[42.0,37.0],[43.0,37.0],[44.0,37.0],[45.0,37.0],[46.0,37.0],[47.0,37.0],[48.0,37.0],[49.0,37.0],[50.0,37.0],[51.0,37.0]],[[0.0,38.0],[1.0,38.0],[2.0,38.0],[3.0,38.0],[4.0,38.0],[5.0,38.0],[6.0,38.0],[7.0,38.0],[8.0,38.0],[9.0,38.0],[10.0,38.0],[11.0,38.0],[12.0,38.0],[13.0,38.0],[14.0,38.0],[15.0,38.0],[16.0,38.0],[17.0,38.0],[18.0,38.0],[19.0,38.0],[20.0,38.0],[21.0,38.0],[22.0,38.0],[23.0,38.0],[24.0,38.0],[25.0,38.0],[26.0,38.0],[27.0,38.0],[28.0,38.0],[29.0,38.0],[30.0,38.0],[31.0,38.0],[32.0,38.0],[33.0,38.0],[34.0,38.0],[35.0,38.0],[36.0,38.0],[37.0,38.0],[38.0,38.0],[39.0,38.0],[40.0,38.0],[41.0,38.0],[42.0,38.0],[43.0,38.0],[44.0,38.0],[45.0,38.0],[46.0,38.0],[47.0,38.0],[48.0,38.0],[49.0,38.0],[50.0,38.0],[51.0,38.0]],[[0.0,39.0],[1.0,39.0],[2.0,39.0],[3.0,39.0],[4.0,39.0],[5.0,39.0],[6.0,39.0],[7.0,39.0],[8.0,39.0],[9.0,39.0],[10.0,39.0],[11.0,39.0],[12.0,39.0],[13.0,39.0],[14.0,39.0],[15.0,39.0],[16.0,39.0],[17.0,39.0],[18.0,39.0],[19.0,39.0],[20.0,39.0],[21.0,39.0],[22.0,39.0],[23.0,39.0],[24.0,39.0],[25.0,39.0],[26.0,39.0],[27.0,39.0],[28.0,39.0],[29.0,39.0],[30.0,39.0],[31.0,39.0],[32.0,39.0],[33.0,39.0],[34.0,39.0],[35.0,39.0],[36.0,39.0],[37.0,39.0],[38.0,39.0],[39.0,39.0],[40.0,39.0],[41.0,39.0],[42.0,39.0],[43.0,39.0],[44.0,39.0],[45.0,39.0],[46.0,39.0],[47.0,39.0],[48.0,39.0],[49.0,39.0],[50.0,39.0],[51.0,39.0]],[[0.0,40.0],[1.0,40.0],[2.0,40.0],[3.0,40.0],[4.0,40.0],[5.0,40.0],[6.0,40.0],[7.0,40.0],[8.0,40.0],[9.0,40.0],[10.0,40.0],[11.0,40.0],[12.0,40.0],[13.0,40.0],[14.0,40.0],[15.0,40.0],[16.0,40.0],[17.0,40.0],[18.0,40.0],[19.0,40.0],[20.0,40.0],[21.0,40.0],[22.0,40.0],[23.0,40.0],[24.0,40.0],[25.0,40.0],[26.0,40.0],[27.0,40.0],[28.0,40.0],[29.0,40.0],[30.0,40.0],[31.0,40.0],[32.0,40.0],[33.0,40.0],[34.0,40.0],[35.0,40.0],[36.0,40.0],[37.0,40.0],[38.0,40.0],[39.0,40.0],[40.0,40.0],[41.0,40.0],[42.0,40.0],[43.0,40.0],[44.0,40.0],[45.0,40.0],[46.0,40.0],[47.0,40.0],[48.0,40.0],[49.0,40.0],[50.0,40.0],[51.0,40.0]],[[0.0,41.0],[1.0,41.0],[2.0,41.0],[3.0,41.0],[4.0,41.0],[5.0,41.0],[6.0,41.0],[7.0,41.0],[8.0,41.0],[9.0,41.0],[10.0,41.0],[11.0,41.0],[12.0,41.0],[13.0,41.0],[14.0,41.0],[15.0,41.0],[16.0,41.0],[17.0,41.0],[18.0,41.0],[19.0,41.0],[20.0,41.0],[21.0,41.0],[22.0,41.0],[23.0,41.0],[24.0,41.0],[25.0,41.0],[26.0,41.0],[27.0,41.0],[28.0,41.0],[29.0,41.0],[30.0,41.0],[31.0,41.0],[32.0,41.0],[33.0,41.0],[34.0,41.0],[35.0,41.0],[36.0,41.0],[37.0,41.0],[38.0,41.0],[39.0,41.0],[40.0,41.0],[41.0,41.0],[42.0,41.0],[43.0,41.0],[44.0,41.0],[45.0,41.0],[46.0,41.0],[47.0,41.0],[48.0,41.0],[49.0,41.0],[50.0,41.0],[51.0,41.0]],[[0.0,42.0],[1.0,42.0],[2.0,42.0],[3.0,42.0],[4.0,42.0],[5.0,42.0],[6.0,42.0],[7.0,42.0],[8.0,42.0],[9.0,42.0],[10.0,42.0],[11.0,42.0],[12.0,42.0],[13.0,42.0],[14.0,42.0],[15.0,42.0],[16.0,42.0],[17.0,42.0],[18.0,42.0],[19.0,42.0],[20.0,42.0],[21.0,42.0],[22.0,42.0],[23.0,42.0],[24.0,42.0],[25.0,42.0],[26.0,42.0],[27.0,42.0],[28.0,42.0],[29.0,42.0],[30.0,42.0],[31.0,42.0],[32.0,42.0],[33.0,42.0],[34.0,42.0],[35.0,42.0],[36.0,42.0],[37.0,42.0],[38.0,42.0],[39.0,42.0],[40.0,42.0],[41.0,42.0],[42.0,42.0],[43.0,42.0],[44.0,42.0],[45.0,42.0],[46.0,42.0],[47.0,42.0],[48.0,42.0],[49.0,42.0],[50.0,42.0],[51.0,42.0]],[[0.0,43.0],[1.0,43.0],[2.0,43.0],[3.0,43.0],[4.0,43.0],[5.0,43.0],[6.0,43.0],[7.0,43.0],[8.0,43.0],[9.0,43.0],[10.0,43.0],[11.0,43.0],[12.0,43.0],[13.0,43.0],[14.0,43.0],[15.0,43.0],[16.0,43.0],[17.0,43.0],[18.0,43.0],[19.0,43.0],[20.0,43.0],[21.0,43.0],[22.0,43.0],[23.0,43.0],[24.0,43.0],[25.0,43.0],[26.0,43.0],[27.0,43.0],[28.0,43.0],[29.0,43.0],[30.0,43.0],[31.0,43.0],[32.0,43.0],[33.0,43.0],[34.0,43.0],[35.0,43.0],[36.0,43.0],[37.0,43.0],[38.0,43.0],[39.0,43.0],[40.0,43.0],[41.0,43.0],[42.0,43.0],[43.0,43.0],[44.0,43.0],[45.0,43.0],[46.0,43.0],[47.0,43.0],[48.0,43.0],[49.0,43.0],[50.0,43.0],[51.0,43.0]],[[0.0,44.0],[1.0,44.0],[2.0,44.0],[3.0,44.0],[4.0,44.0],[5.0,44.0],[6.0,44.0],[7.0,44.0],[8.0,44.0],[9.0,44.0],[10.0,44.0],[11.0,44.0],[12.0,44.0],[13.0,44.0],[14.0,44.0],[15.0,44.0],[16.0,44.0],[17.0,44.0],[18.0,44.0],[19.0,44.0],[20.0,44.0],[21.0,44.0],[22.0,44.0],[23.0,44.0],[24.0,44.0],[25.0,44.0],[26.0,44.0],[27.0,44.0],[28.0,44.0],[29.0,44.0],[30.0,44.0],[31.0,44.0],[32.0,44.0],[33.0,44.0],[34.0,44.0],[35.0,44.0],[36.0,44.0],[37.0,44.0],[38.0,44.0],[39.0,44.0],[40.0,44.0],[41.0,44.0],[42.0,44.0],[43.0,44.0],[44.0,44.0],[45.0,44.0],[46.0,44.0],[47.0,44.0],[48.0,44.0],[49.0,44.0],[50.0,44.0],[51.0,44.0]],[[0.0,45.0],[1.0,45.0],[2.0,45.0],[3.0,45.0],[4.0,45.0],[5.0,45.0],[6.0,45.0],[7.0,45.0],[8.0,45.0],[9.0,45.0],[10.0,45.0],[11.0,45.0],[12.0,45.0],[13.0,45.0],[14.0,45.0],[15.0,45.0],[16.0,45.0],[17.0,45.0],[18.0,45.0],[19.0,45.0],[20.0,45.0],[21.0,45.0],[22.0,45.0],[23.0,45.0],[24.0,45.0],[25.0,45.0],[26.0,45.0],[27.0,45.0],[28.0,45.0],[29.0,45.0],[30.0,45.0],[31.0,45.0],[32.0,45.0],[33.0,45.0],[34.0,45.0],[35.0,45.0],[36.0,45.0],[37.0,45.0],[38.0,45.0],[39.0,45.0],[40.0,45.0],[41.0,45.0],[42.0,45.0],[43.0,45.0],[44.0,45.0],[45.0,45.0],[46.0,45.0],[47.0,45.0],[48.0,45.0],[49.0,45.0],[50.0,45.0],[51.0,45.0]],[[0.0,46.0],[1.0,46.0],[2.0,46.0],[3.0,46.0],[4.0,46.0],[5.0,46.0],[6.0,46.0],[7.0,46.0],[8.0,46.0],[9.0,46.0],[10.0,46.0],[11.0,46.0],[12.0,46.0],[13.0,46.0],[14.0,46.0],[15.0,46.0],[16.0,46.0],[17.0,46.0],[18.0,46.0],[19.0,46.0],[20.0,46.0],[21.0,46.0],[22.0,46.0],[23.0,46.0],[24.0,46.0],[25.0,46.0],[26.0,46.0],[27.0,46.0],[28.0,46.0],[29.0,46.0],[30.0,46.0],[31.0,46.0],[32.0,46.0],[33.0,46.0],[34.0,46.0],[35.0,46.0],[36.0,46.0],[37.0,46.0],[38.0,46.0],[39.0,46.0],[40.0,46.0],[41.0,46.0],[42.0,46.0],[43.0,46.0],[44.0,46.0],[45.0,46.0],[46.0,46.0],[47.0,46.0],[48.0,46.0],[49.0,46.0],[50.0,46.0],[51.0,46.0]],[[0.0,47.0],[1.0,47.0],[2.0,47.0],[3.0,47.0],[4.0,47.0],[5.0,47.0],[6.0,47.0],[7.0,47.0],[8.0,47.0],[9.0,47.0],[10.0,47.0],[11.0,47.0],[12.0,47.0],[13.0,47.0],[14.0,47.0],[15.0,47.0],[16.0,47.0],[17.0,47.0],[18.0,47.0],[19.0,47.0],[20.0,47.0],[21.0,47.0],[22.0,47.0],[23.0,47.0],[24.0,47.0],[25.0,47.0],[26.0,47.0],[27.0,47.0],[28.0,47.0],[29.0,47.0],[30.0,47.0],[31.0,47.0],[32.0,47.0],[33.0,47.0],[34.0,47.0],[35.0,47.0],[36.0,47.0],[37.0,47.0],[38.0,47.0],[39.0,47.0],[40.0,47.0],[41.0,47.0],[42.0,47.0],[43.0,47.0],[44.0,47.0],[45.0,47.0],[46.0,47.0],[47.0,47.0],[48.0,47.0],[49.0,47.0],[50.0,47.0],[51.0,47.0]],[[0.0,48.0],[1.0,48.0],[2.0,48.0],[3.0,48.0],[4.0,48.0],[5.0,48.0],[6.0,48.0],[7.0,48.0],[8.0,48.0],[9.0,48.0],[10.0,48.0],[11.0,48.0],[12.0,48.0],[13.0,48.0],[14.0,48.0],[15.0,48.0],[16.0,48.0],[17.0,48.0],[18.0,48.0],[19.0,48.0],[20.0,48.0],[21.0,48.0],[22.0,48.0],[23.0,48.0],[24.0,48.0],[25.0,48.0],[26.0,48.0],[27.0,48.0],[28.0,48.0],[29.0,48.0],[30.0,48.0],[31.0,48.0],[32.0,48.0],[33.0,48.0],[34.0,48.0],[35.0,48.0],[36.0,48.0],[37.0,48.0],[38.0,48.0],[39.0,48.0],[40.0,48.0],[41.0,48.0],[42.0,48.0],[43.0,48.0],[44.0,48.0],[45.0,48.0],[46.0,48.0],[47.0,48.0],[48.0,48.0],[49.0,48.0],[50.0,48.0],[51.0,48.0]],[[0.0,49.0],[1.0,49.0],[2.0,49.0],[3.0,49.0],[4.0,49.0],[5.0,49.0],[6.0,49.0],[7.0,49.0],[8.0,49.0],[9.0,49.0],[10.0,49.0],[11.0,49.0],[12.0,49.0],[13.0,49.0],[14.0,49.0],[15.0,49.0],[16.0,49.0],[17.0,49.0],[18.0,49.0],[19.0,49.0],[20.0,49.0],[21.0,49.0],[22.0,49.0],[23.0,49.0],[24.0,49.0],[25.0,49.0],[26.0,49.0],[27.0,49.0],[28.0,49.0],[29.0,49.0],[30.0,49.0],[31.0,49.0],[32.0,49.0],[33.0,49.0],[34.0,49.0],[35.0,49.0],[36.0,49.0],[37.0,49.0],[38.0,49.0],[39.0,49.0],[40.0,49.0],[41.0,49.0],[42.0,49.0],[43.0,49.0],[44.0,49.0],[45.0,49.0],[46.0,49.0],[47.0,49.0],[48.0,49.0],[49.0,49.0],[50.0,49.0],[51.0,49.0]],[[0.0,50.0],[1.0,50.0],[2.0,50.0],[3.0,50.0],[4.0,50.0],[5.0,50.0],[6.0,50.0],[7.0,50.0],[8.0,50.0],[9.0,50.0],[10.0,50.0],[11.0,50.0],[12.0,50.0],[13.0,50.0],[14.0,50.0],[15.0,50.0],[16.0,50.0],[17.0,50.0],[18.0,50.0],[19.0,50.0],[20.0,50.0],[21.0,50.0],[22.0,50.0],[23.0,50.0],[24.0,50.0],[25.0,50.0],[26.0,50.0],[27.0,50.0],[28.0,50.0],[29.0,50.0],[30.0,50.0],[31.0,50.0],[32.0,50.0],[33.0,50.0],[34.0,50.0],[35.0,50.0],[36.0,50.0],[37.0,50.0],[38.0,50.0],[39.0,50.0],[40.0,50.0],[41.0,50.0],[42.0,50.0],[43.0,50.0],[44.0,50.0],[45.0,50.0],[46.0,50.0],[47.0,50.0],[48.0,50.0],[49.0,50.0],[50.0,50.0],[51.0,50.0]],[[0.0,51.0],[1.0,51.0],[2.0,51.0],[3.0,51.0],[4.0,51.0],[5.0,51.0],[6.0,51.0],[7.0,51.0],[8.0,51.0],[9.0,51.0],[10.0,51.0],[11.0,51.0],[12.0,51.0],[13.0,51.0],[14.0,51.0],[15.0,51.0],[16.0,51.0],[17.0,51.0],[18.0,51.0],[19.0,51.0],[20.0,51.0],[21.0,51.0],[22.0,51.0],[23.0,51.0],[24.0,51.0],[25.0,51.0],[26.0,51.0],[27.0,51.0],[28.0,51.0],[29.0,51.0],[30.0,51.0],[31.0,51.0],[32.0,51.0],[33.0,51.0],[34.0,51.0],[35.0,51.0],[36.0,51.0],[37.0,51.0],[38.0,51.0],[39.0,51.0],[40.0,51.0],[41.0,51.0],[42.0,51.0],[43.0,51.0],[44.0,51.0],[45.0,51.0],[46.0,51.0],[47.0,51.0],[48.0,51.0],[49.0,51.0],[50.0,51.0],[51.0,51.0]]]]], dtype=torch.float, device='cpu')
        output_module_11 = self.module_11(input=args[0])
        output_module_12 = self.module_12(input=output_module_11, dim=2)
        output_module_12 = self.module_13(data=output_module_12, dtype=torch.int, device='cpu')
        output_module_14 = self.module_14(output_module_11)
        output_module_14 = self.module_15(output_module_14)
        output_module_14 = self.module_16(output_module_14)
        output_module_14 = self.module_17(output_module_14)
        output_module_18 = self.module_18(output_module_14)
        output_module_18 = self.module_19(output_module_18)
        output_module_18 = self.module_20(output_module_18)
        output_module_18 = self.module_21(output_module_18)
        output_module_18 = self.module_22(input=output_module_18, other=output_module_14, alpha=1)
        output_module_18 = self.module_23(output_module_18)
        output_module_18 = self.module_24(output_module_18)
        output_module_25 = self.module_25(output_module_18)
        output_module_25 = self.module_26(output_module_25)
        output_module_25 = self.module_27(output_module_25)
        output_module_25 = self.module_28(output_module_25)
        output_module_25 = self.module_29(input=output_module_25, other=output_module_18, alpha=1)
        output_module_30 = self.module_30(output_module_25)
        output_module_30 = self.module_31(output_module_30)
        output_module_30 = self.module_32(output_module_30)
        output_module_30 = self.module_33(output_module_30)
        output_module_30 = self.module_34(input=output_module_30, other=output_module_25, alpha=1)
        output_module_30 = self.module_35(output_module_30)
        output_module_30 = self.module_36(output_module_30)
        output_module_37 = self.module_37(output_module_30)
        output_module_37 = self.module_38(output_module_37)
        output_module_37 = self.module_39(output_module_37)
        output_module_37 = self.module_40(output_module_37)
        output_module_37 = self.module_41(input=output_module_37, other=output_module_30, alpha=1)
        output_module_42 = self.module_42(output_module_37)
        output_module_42 = self.module_43(output_module_42)
        output_module_42 = self.module_44(output_module_42)
        output_module_42 = self.module_45(output_module_42)
        output_module_42 = self.module_46(input=output_module_42, other=output_module_37, alpha=1)
        output_module_47 = self.module_47(output_module_42)
        output_module_47 = self.module_48(output_module_47)
        output_module_47 = self.module_49(output_module_47)
        output_module_47 = self.module_50(output_module_47)
        output_module_47 = self.module_51(input=output_module_47, other=output_module_42, alpha=1)
        output_module_52 = self.module_52(output_module_47)
        output_module_52 = self.module_53(output_module_52)
        output_module_52 = self.module_54(output_module_52)
        output_module_52 = self.module_55(output_module_52)
        output_module_52 = self.module_56(input=output_module_52, other=output_module_47, alpha=1)
        output_module_57 = self.module_57(output_module_52)
        output_module_57 = self.module_58(output_module_57)
        output_module_57 = self.module_59(output_module_57)
        output_module_57 = self.module_60(output_module_57)
        output_module_57 = self.module_61(input=output_module_57, other=output_module_52, alpha=1)
        output_module_62 = self.module_62(output_module_57)
        output_module_62 = self.module_63(output_module_62)
        output_module_62 = self.module_64(output_module_62)
        output_module_62 = self.module_65(output_module_62)
        output_module_62 = self.module_66(input=output_module_62, other=output_module_57, alpha=1)
        output_module_67 = self.module_67(output_module_62)
        output_module_67 = self.module_68(output_module_67)
        output_module_67 = self.module_69(output_module_67)
        output_module_67 = self.module_70(output_module_67)
        output_module_67 = self.module_71(input=output_module_67, other=output_module_62, alpha=1)
        output_module_72 = self.module_72(output_module_67)
        output_module_72 = self.module_73(output_module_72)
        output_module_72 = self.module_74(output_module_72)
        output_module_72 = self.module_75(output_module_72)
        output_module_72 = self.module_76(input=output_module_72, other=output_module_67, alpha=1)
        output_module_77 = self.module_77(output_module_72)
        output_module_77 = self.module_78(output_module_77)
        output_module_79 = self.module_79(output_module_77)
        output_module_79 = self.module_80(output_module_79)
        output_module_79 = self.module_81(output_module_79)
        output_module_79 = self.module_82(output_module_79)
        output_module_79 = self.module_83(input=output_module_79, other=output_module_77, alpha=1)
        output_module_84 = self.module_84(output_module_79)
        output_module_84 = self.module_85(output_module_84)
        output_module_84 = self.module_86(output_module_84)
        output_module_84 = self.module_87(output_module_84)
        output_module_84 = self.module_88(input=output_module_84, other=output_module_79, alpha=1)
        output_module_89 = self.module_89(output_module_84)
        output_module_89 = self.module_90(output_module_89)
        output_module_89 = self.module_91(output_module_89)
        output_module_89 = self.module_92(output_module_89)
        output_module_89 = self.module_93(input=output_module_89, other=output_module_84, alpha=1)
        output_module_94 = self.module_94(output_module_89)
        output_module_94 = self.module_95(output_module_94)
        output_module_94 = self.module_96(output_module_94)
        output_module_94 = self.module_97(output_module_94)
        output_module_94 = self.module_98(input=output_module_94, other=output_module_89, alpha=1)
        output_module_99 = self.module_99(output_module_94)
        output_module_99 = self.module_100(output_module_99)
        output_module_99 = self.module_101(output_module_99)
        output_module_99 = self.module_102(output_module_99)
        output_module_99 = self.module_103(input=output_module_99, other=output_module_94, alpha=1)
        output_module_104 = self.module_104(output_module_99)
        output_module_104 = self.module_105(output_module_104)
        output_module_104 = self.module_106(output_module_104)
        output_module_104 = self.module_107(output_module_104)
        output_module_104 = self.module_108(input=output_module_104, other=output_module_99, alpha=1)
        output_module_109 = self.module_109(output_module_104)
        output_module_109 = self.module_110(output_module_109)
        output_module_109 = self.module_111(output_module_109)
        output_module_109 = self.module_112(output_module_109)
        output_module_109 = self.module_113(input=output_module_109, other=output_module_104, alpha=1)
        output_module_114 = self.module_114(output_module_109)
        output_module_114 = self.module_115(output_module_114)
        output_module_114 = self.module_116(output_module_114)
        output_module_114 = self.module_117(output_module_114)
        output_module_114 = self.module_118(input=output_module_114, other=output_module_109, alpha=1)
        output_module_119 = self.module_119(output_module_114)
        output_module_119 = self.module_120(output_module_119)
        output_module_121 = self.module_121(output_module_119)
        output_module_121 = self.module_122(output_module_121)
        output_module_121 = self.module_123(output_module_121)
        output_module_121 = self.module_124(output_module_121)
        output_module_121 = self.module_125(input=output_module_121, other=output_module_119, alpha=1)
        output_module_126 = self.module_126(output_module_121)
        output_module_126 = self.module_127(output_module_126)
        output_module_126 = self.module_128(output_module_126)
        output_module_126 = self.module_129(output_module_126)
        output_module_126 = self.module_130(input=output_module_126, other=output_module_121, alpha=1)
        output_module_131 = self.module_131(output_module_126)
        output_module_131 = self.module_132(output_module_131)
        output_module_131 = self.module_133(output_module_131)
        output_module_131 = self.module_134(output_module_131)
        output_module_131 = self.module_135(input=output_module_131, other=output_module_126, alpha=1)
        output_module_136 = self.module_136(output_module_131)
        output_module_136 = self.module_137(output_module_136)
        output_module_136 = self.module_138(output_module_136)
        output_module_136 = self.module_139(output_module_136)
        output_module_136 = self.module_140(input=output_module_136, other=output_module_131, alpha=1)
        output_module_136 = self.module_141(output_module_136)
        output_module_136 = self.module_142(output_module_136)
        output_module_136 = self.module_143(output_module_136)
        output_module_136 = self.module_144(output_module_136)
        output_module_136 = self.module_145(output_module_136)
        output_module_136 = self.module_146(output_module_136)
        output_module_136 = self.module_147(output_module_136)
        output_module_136 = self.module_148(output_module_136)
        output_module_136 = self.module_149(output_module_136)
        output_module_136 = self.module_150(output_module_136)
        output_module_151 = self.module_151(output_module_136)
        output_module_151 = self.module_152(output_module_151)
        output_module_151 = self.module_153(output_module_151)
        output_module_154 = self.module_154(input=output_module_151, dim=2)
        output_module_154 = self.module_155(data=output_module_154, dtype=torch.int, device='cpu')
        output_module_156 = self.module_156({'self': output_module_12,'other': output_module_154,'rounding_mode': 'trunc'})
        output_module_157 = self.module_157(input=output_module_151, dim=0)
        output_module_158 = self.module_158(input=output_module_151, dim=2)
        output_module_159 = self.module_159(input=output_module_151, dim=3)
        output_module_160 = self.module_160(input=output_module_151, shape=[output_module_157,3,85,output_module_158,output_module_159])
        output_module_160 = self.module_161(dims=[0,1,3,4,2], input=output_module_160)
        output_module_160 = self.module_162(output_module_160)
        output_module_163 = self.module_163(input=output_module_160, dim=[4], start=[0], end=[2], step=[1])
        output_module_163 = self.module_164(output_module_163)
        output_module_163 = self.module_165(input=output_module_163, other=output_module_4, alpha=1)
        output_module_163 = self.module_166(input=output_module_163, other=output_module_156)
        output_module_160[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2:1] = output_module_163
        output_module_167 = self.module_167(input=output_module_160, dim=[4], start=[2], end=[4], step=[1])
        output_module_167 = self.module_168(input=output_module_167)
        output_module_167 = self.module_169(input=output_module_167, other=self.module_list_82_yolo_82_anchor_grid)
        output_module_160[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,2:4:1] = output_module_167
        output_module_170 = self.module_170(input=output_module_160, dim=[4], start=[4], end=[9223372036854775807], step=[1])
        output_module_170 = self.module_171(output_module_170)
        output_module_160[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,4:9223372036854775807:1] = output_module_170
        output_module_172 = self.module_172(input=output_module_160, shape=[output_module_157,-1,85])
        output_module_173 = self.module_173(dim=1, tensors=[output_module_136])
        output_module_174 = self.module_174(input=output_module_173, dim=1)
        output_module_174 = self.module_175(data=output_module_174, dtype=torch.int, device='cpu')
        output_module_174 = self.module_176({'self': output_module_174,'other': output_module_5,'rounding_mode': 'trunc'})
        output_module_177 = self.module_177(input=output_module_174, other=output_module_0)
        output_module_177 = self.module_178(input=output_module_177)
        output_module_179 = self.module_179(input=output_module_174)
        output_module_180 = self.module_180(input=output_module_173, dim=[0,1], start=[0,output_module_177], end=[9223372036854775807,output_module_179], step=[1,1])
        output_module_180 = self.module_181(output_module_180)
        output_module_180 = self.module_182(output_module_180)
        output_module_180 = self.module_183(input=output_module_180, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_180 = self.module_184(dim=1, tensors=[output_module_180,output_module_114])
        output_module_185 = self.module_185(input=output_module_180, dim=1)
        output_module_185 = self.module_186(data=output_module_185, dtype=torch.int, device='cpu')
        output_module_185 = self.module_187({'self': output_module_185,'other': output_module_6,'rounding_mode': 'trunc'})
        output_module_188 = self.module_188(input=output_module_185, other=output_module_1)
        output_module_188 = self.module_189(input=output_module_188)
        output_module_190 = self.module_190(input=output_module_185)
        output_module_191 = self.module_191(input=output_module_180, dim=[0,1], start=[0,output_module_188], end=[9223372036854775807,output_module_190], step=[1,1])
        output_module_191 = self.module_192(output_module_191)
        output_module_191 = self.module_193(output_module_191)
        output_module_191 = self.module_194(output_module_191)
        output_module_191 = self.module_195(output_module_191)
        output_module_191 = self.module_196(output_module_191)
        output_module_191 = self.module_197(output_module_191)
        output_module_191 = self.module_198(output_module_191)
        output_module_191 = self.module_199(output_module_191)
        output_module_191 = self.module_200(output_module_191)
        output_module_191 = self.module_201(output_module_191)
        output_module_202 = self.module_202(output_module_191)
        output_module_202 = self.module_203(output_module_202)
        output_module_202 = self.module_204(output_module_202)
        output_module_205 = self.module_205(input=output_module_202, dim=2)
        output_module_205 = self.module_206(data=output_module_205, dtype=torch.int, device='cpu')
        output_module_207 = self.module_207({'self': output_module_12,'other': output_module_205,'rounding_mode': 'trunc'})
        output_module_208 = self.module_208(input=output_module_202, dim=0)
        output_module_209 = self.module_209(input=output_module_202, dim=2)
        output_module_210 = self.module_210(input=output_module_202, dim=3)
        output_module_211 = self.module_211(input=output_module_202, shape=[output_module_208,3,85,output_module_209,output_module_210])
        output_module_211 = self.module_212(dims=[0,1,3,4,2], input=output_module_211)
        output_module_211 = self.module_213(output_module_211)
        output_module_214 = self.module_214(input=output_module_211, dim=[4], start=[0], end=[2], step=[1])
        output_module_214 = self.module_215(output_module_214)
        output_module_214 = self.module_216(input=output_module_214, other=output_module_7, alpha=1)
        output_module_214 = self.module_217(input=output_module_214, other=output_module_207)
        output_module_211[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2:1] = output_module_214
        output_module_218 = self.module_218(input=output_module_211, dim=[4], start=[2], end=[4], step=[1])
        output_module_218 = self.module_219(input=output_module_218)
        output_module_218 = self.module_220(input=output_module_218, other=self.module_list_94_yolo_94_anchor_grid)
        output_module_211[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,2:4:1] = output_module_218
        output_module_221 = self.module_221(input=output_module_211, dim=[4], start=[4], end=[9223372036854775807], step=[1])
        output_module_221 = self.module_222(output_module_221)
        output_module_211[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,4:9223372036854775807:1] = output_module_221
        output_module_223 = self.module_223(input=output_module_211, shape=[output_module_208,-1,85])
        output_module_224 = self.module_224(dim=1, tensors=[output_module_191])
        output_module_225 = self.module_225(input=output_module_224, dim=1)
        output_module_225 = self.module_226(data=output_module_225, dtype=torch.int, device='cpu')
        output_module_225 = self.module_227({'self': output_module_225,'other': output_module_8,'rounding_mode': 'trunc'})
        output_module_228 = self.module_228(input=output_module_225, other=output_module_2)
        output_module_228 = self.module_229(input=output_module_228)
        output_module_230 = self.module_230(input=output_module_225)
        output_module_231 = self.module_231(input=output_module_224, dim=[0,1], start=[0,output_module_228], end=[9223372036854775807,output_module_230], step=[1,1])
        output_module_231 = self.module_232(output_module_231)
        output_module_231 = self.module_233(output_module_231)
        output_module_231 = self.module_234(input=output_module_231, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_231 = self.module_235(dim=1, tensors=[output_module_231,output_module_72])
        output_module_236 = self.module_236(input=output_module_231, dim=1)
        output_module_236 = self.module_237(data=output_module_236, dtype=torch.int, device='cpu')
        output_module_236 = self.module_238({'self': output_module_236,'other': output_module_9,'rounding_mode': 'trunc'})
        output_module_239 = self.module_239(input=output_module_236, other=output_module_3)
        output_module_239 = self.module_240(input=output_module_239)
        output_module_241 = self.module_241(input=output_module_236)
        output_module_242 = self.module_242(input=output_module_231, dim=[0,1], start=[0,output_module_239], end=[9223372036854775807,output_module_241], step=[1,1])
        output_module_242 = self.module_243(output_module_242)
        output_module_242 = self.module_244(output_module_242)
        output_module_242 = self.module_245(output_module_242)
        output_module_242 = self.module_246(output_module_242)
        output_module_242 = self.module_247(output_module_242)
        output_module_242 = self.module_248(output_module_242)
        output_module_242 = self.module_249(output_module_242)
        output_module_242 = self.module_250(output_module_242)
        output_module_242 = self.module_251(output_module_242)
        output_module_242 = self.module_252(output_module_242)
        output_module_242 = self.module_253(output_module_242)
        output_module_242 = self.module_254(output_module_242)
        output_module_242 = self.module_255(output_module_242)
        output_module_256 = self.module_256(input=output_module_242, dim=2)
        output_module_256 = self.module_257(data=output_module_256, dtype=torch.int, device='cpu')
        output_module_258 = self.module_258({'self': output_module_12,'other': output_module_256,'rounding_mode': 'trunc'})
        output_module_259 = self.module_259(input=output_module_242, dim=0)
        output_module_260 = self.module_260(input=output_module_242, dim=2)
        output_module_261 = self.module_261(input=output_module_242, dim=3)
        output_module_262 = self.module_262(input=output_module_242, shape=[output_module_259,3,85,output_module_260,output_module_261])
        output_module_262 = self.module_263(dims=[0,1,3,4,2], input=output_module_262)
        output_module_262 = self.module_264(output_module_262)
        output_module_265 = self.module_265(input=output_module_262, dim=[4], start=[0], end=[2], step=[1])
        output_module_265 = self.module_266(output_module_265)
        output_module_265 = self.module_267(input=output_module_265, other=output_module_10, alpha=1)
        output_module_265 = self.module_268(input=output_module_265, other=output_module_258)
        output_module_262[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2:1] = output_module_265
        output_module_269 = self.module_269(input=output_module_262, dim=[4], start=[2], end=[4], step=[1])
        output_module_269 = self.module_270(input=output_module_269)
        output_module_269 = self.module_271(input=output_module_269, other=self.module_list_106_yolo_106_anchor_grid)
        output_module_262[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,2:4:1] = output_module_269
        output_module_272 = self.module_272(input=output_module_262, dim=[4], start=[4], end=[9223372036854775807], step=[1])
        output_module_272 = self.module_273(output_module_272)
        output_module_262[0:2147483647:1,0:2147483647:1,0:2147483647:1,0:2147483647:1,4:9223372036854775807:1] = output_module_272
        output_module_274 = self.module_274(input=output_module_262, shape=[output_module_259,-1,85])
        output_module_172 = self.module_275(dim=1, tensors=[output_module_172,output_module_223,output_module_274])
        return output_module_172
