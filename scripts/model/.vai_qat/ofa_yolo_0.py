# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ofa_yolo_0(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ofa_yolo_0, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ofa_yolo_0::input_0
        self.module_1 = py_nndct.nn.quant_input() #ofa_yolo_0::ofa_yolo_0/QuantStub[quant]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=[6, 6], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_0]/input.3
        self.module_3 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_2]/input.7
        self.module_4 = py_nndct.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_3]/input.9
        self.module_5 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_5]/input.13
        self.module_6 = py_nndct.nn.Conv2d(in_channels=96, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_6]/input.15
        self.module_7 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_8]/input.19
        self.module_8 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_9]/input.21
        self.module_9 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_11]/input.25
        self.module_10 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_12]/input.27
        self.module_11 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_14]/8822
        self.module_12 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_244]/input.31
        self.module_13 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_15]/input.33
        self.module_14 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_17]/input.37
        self.module_15 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_18]/input.39
        self.module_16 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_20]/8878
        self.module_17 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_245]/8880
        self.module_18 = py_nndct.nn.Conv2d(in_channels=96, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_21]/input.43
        self.module_19 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_23]/8907
        self.module_20 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_245]/input.47
        self.module_21 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_24]/input.49
        self.module_22 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_26]/input.53
        self.module_23 = py_nndct.nn.Conv2d(in_channels=96, out_channels=192, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_27]/input.55
        self.module_24 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_29]/input.59
        self.module_25 = py_nndct.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_30]/input.61
        self.module_26 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_32]/input.65
        self.module_27 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_33]/input.67
        self.module_28 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_35]/input.71
        self.module_29 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_36]/input.73
        self.module_30 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_38]/9045
        self.module_31 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_246]/input.77
        self.module_32 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_39]/input.79
        self.module_33 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_41]/input.83
        self.module_34 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_42]/input.85
        self.module_35 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_44]/9101
        self.module_36 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_247]/input.89
        self.module_37 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_45]/input.91
        self.module_38 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_47]/input.95
        self.module_39 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_48]/input.97
        self.module_40 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_50]/9157
        self.module_41 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_248]/input.101
        self.module_42 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_51]/input.103
        self.module_43 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_53]/input.107
        self.module_44 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_54]/input.109
        self.module_45 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_56]/9213
        self.module_46 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_249]/9215
        self.module_47 = py_nndct.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_57]/input.113
        self.module_48 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_59]/9242
        self.module_49 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_249]/input.117
        self.module_50 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_60]/input.119
        self.module_51 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_62]/input.123
        self.module_52 = py_nndct.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_63]/input.125
        self.module_53 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_65]/input.129
        self.module_54 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_66]/input.131
        self.module_55 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_68]/input.135
        self.module_56 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_69]/input.137
        self.module_57 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_71]/input.141
        self.module_58 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_72]/input.143
        self.module_59 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_74]/9380
        self.module_60 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_250]/input.147
        self.module_61 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_75]/input.149
        self.module_62 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_77]/input.153
        self.module_63 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_78]/input.155
        self.module_64 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_80]/9436
        self.module_65 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_251]/input.159
        self.module_66 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_81]/input.161
        self.module_67 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_83]/input.165
        self.module_68 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_84]/input.167
        self.module_69 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_86]/9492
        self.module_70 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_252]/input.171
        self.module_71 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_87]/input.173
        self.module_72 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_89]/input.177
        self.module_73 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_90]/input.179
        self.module_74 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_92]/9548
        self.module_75 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_253]/input.183
        self.module_76 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_93]/input.185
        self.module_77 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_95]/input.189
        self.module_78 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_96]/input.191
        self.module_79 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_98]/9604
        self.module_80 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_254]/input.195
        self.module_81 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_99]/input.197
        self.module_82 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_101]/input.201
        self.module_83 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_102]/input.203
        self.module_84 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_104]/9660
        self.module_85 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_255]/9662
        self.module_86 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_105]/input.207
        self.module_87 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_107]/9689
        self.module_88 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_255]/input.211
        self.module_89 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_108]/input.213
        self.module_90 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_110]/input.217
        self.module_91 = py_nndct.nn.Conv2d(in_channels=384, out_channels=768, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_111]/input.219
        self.module_92 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_113]/input.223
        self.module_93 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_114]/input.225
        self.module_94 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_116]/input.229
        self.module_95 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_117]/input.231
        self.module_96 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_119]/input.235
        self.module_97 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_120]/input.237
        self.module_98 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_122]/9827
        self.module_99 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_256]/input.241
        self.module_100 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_123]/input.243
        self.module_101 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_125]/input.247
        self.module_102 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_126]/input.249
        self.module_103 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_128]/9883
        self.module_104 = py_nndct.nn.Add() #ofa_yolo_0::ofa_yolo_0/Model[model]/Add[add_257]/9885
        self.module_105 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_129]/input.253
        self.module_106 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_131]/9912
        self.module_107 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_257]/input.257
        self.module_108 = py_nndct.nn.Conv2d(in_channels=768, out_channels=768, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_132]/input.259
        self.module_109 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_134]/input.263
        self.module_110 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_135]/input.265
        self.module_111 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_137]/9969
        self.module_112 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #ofa_yolo_0::ofa_yolo_0/Model[model]/MaxPool2d[module_138]/9983
        self.module_113 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #ofa_yolo_0::ofa_yolo_0/Model[model]/MaxPool2d[module_139]/9997
        self.module_114 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #ofa_yolo_0::ofa_yolo_0/Model[model]/MaxPool2d[module_140]/10011
        self.module_115 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_258]/input.269
        self.module_116 = py_nndct.nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_141]/input.271
        self.module_117 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_143]/input.275
        self.module_118 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_144]/input.277
        self.module_119 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_146]/output_258
        self.module_120 = py_nndct.nn.Interpolate() #ofa_yolo_0::ofa_yolo_0/Model[model]/10073
        self.module_121 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_259]/input.281
        self.module_122 = py_nndct.nn.Conv2d(in_channels=768, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_147]/input.283
        self.module_123 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_149]/input.287
        self.module_124 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_150]/input.289
        self.module_125 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_152]/input.293
        self.module_126 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_153]/input.295
        self.module_127 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_155]/input.299
        self.module_128 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_156]/input.301
        self.module_129 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_158]/input.305
        self.module_130 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_159]/input.307
        self.module_131 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_161]/10211
        self.module_132 = py_nndct.nn.Conv2d(in_channels=768, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_162]/input.311
        self.module_133 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_164]/10238
        self.module_134 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_147]/input.315
        self.module_135 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_165]/input.317
        self.module_136 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_167]/input.321
        self.module_137 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_168]/input.323
        self.module_138 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_170]/output_module_147
        self.module_139 = py_nndct.nn.Interpolate() #ofa_yolo_0::ofa_yolo_0/Model[model]/10300
        self.module_140 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_260]/input.327
        self.module_141 = py_nndct.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_171]/input.329
        self.module_142 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_173]/input.333
        self.module_143 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_174]/input.335
        self.module_144 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_176]/input.339
        self.module_145 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_177]/input.341
        self.module_146 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_179]/input.345
        self.module_147 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_180]/input.347
        self.module_148 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_182]/input.351
        self.module_149 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_183]/input.353
        self.module_150 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_185]/10438
        self.module_151 = py_nndct.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_186]/input.357
        self.module_152 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_188]/10465
        self.module_153 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_171]/input.361
        self.module_154 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_189]/input.363
        self.module_155 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_191]/input.367
        self.module_156 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_192]/input.369
        self.module_157 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_194]/10522
        self.module_158 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_192]/input.373
        self.module_159 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_195]/input.375
        self.module_160 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_197]/input.379
        self.module_161 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_198]/input.381
        self.module_162 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_200]/input.385
        self.module_163 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_201]/input.387
        self.module_164 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_203]/input.391
        self.module_165 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_204]/input.393
        self.module_166 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_206]/input.397
        self.module_167 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_207]/input.399
        self.module_168 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_209]/10660
        self.module_169 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_210]/input.403
        self.module_170 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_212]/10687
        self.module_171 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_195]/input.407
        self.module_172 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_213]/input.409
        self.module_173 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_215]/input.413
        self.module_174 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_216]/input.415
        self.module_175 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_218]/10744
        self.module_176 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_216]/input.419
        self.module_177 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_219]/input.421
        self.module_178 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_221]/input.425
        self.module_179 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_222]/input.427
        self.module_180 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_224]/input.431
        self.module_181 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_225]/input.433
        self.module_182 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_227]/input.437
        self.module_183 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_228]/input.439
        self.module_184 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_230]/input.443
        self.module_185 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_231]/input.445
        self.module_186 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_233]/10882
        self.module_187 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_234]/input.449
        self.module_188 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_236]/10909
        self.module_189 = py_nndct.nn.Cat() #ofa_yolo_0::ofa_yolo_0/Model[model]/Cat[cat_219]/input.453
        self.module_190 = py_nndct.nn.Conv2d(in_channels=768, out_channels=768, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_237]/input.455
        self.module_191 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/LeakyReLU[module_239]/input
        self.module_192 = py_nndct.nn.Conv2d(in_channels=192, out_channels=36, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_240]/ip.1
        self.module_193 = py_nndct.nn.Conv2d(in_channels=384, out_channels=36, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_241]/ip.3
        self.module_194 = py_nndct.nn.Conv2d(in_channels=768, out_channels=36, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ofa_yolo_0::ofa_yolo_0/Model[model]/Conv2d[module_242]/ip
        self.module_195 = py_nndct.nn.dequant_output() #ofa_yolo_0::ofa_yolo_0/DeQuantStub[dequant]/11001
        self.module_196 = py_nndct.nn.dequant_output() #ofa_yolo_0::ofa_yolo_0/DeQuantStub[dequant]/11004
        self.module_197 = py_nndct.nn.dequant_output() #ofa_yolo_0::ofa_yolo_0/DeQuantStub[dequant]/11007

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_6 = self.module_6(output_module_0)
        output_module_6 = self.module_7(output_module_6)
        output_module_8 = self.module_8(output_module_6)
        output_module_8 = self.module_9(output_module_8)
        output_module_8 = self.module_10(output_module_8)
        output_module_8 = self.module_11(output_module_8)
        output_module_12 = self.module_12(input=output_module_6, other=output_module_8, alpha=1)
        output_module_13 = self.module_13(output_module_12)
        output_module_13 = self.module_14(output_module_13)
        output_module_13 = self.module_15(output_module_13)
        output_module_13 = self.module_16(output_module_13)
        output_module_17 = self.module_17(input=output_module_12, other=output_module_13, alpha=1)
        output_module_18 = self.module_18(output_module_0)
        output_module_18 = self.module_19(output_module_18)
        output_module_17 = self.module_20(dim=1, tensors=[output_module_17,output_module_18])
        output_module_17 = self.module_21(output_module_17)
        output_module_17 = self.module_22(output_module_17)
        output_module_17 = self.module_23(output_module_17)
        output_module_17 = self.module_24(output_module_17)
        output_module_25 = self.module_25(output_module_17)
        output_module_25 = self.module_26(output_module_25)
        output_module_27 = self.module_27(output_module_25)
        output_module_27 = self.module_28(output_module_27)
        output_module_27 = self.module_29(output_module_27)
        output_module_27 = self.module_30(output_module_27)
        output_module_31 = self.module_31(input=output_module_25, other=output_module_27, alpha=1)
        output_module_32 = self.module_32(output_module_31)
        output_module_32 = self.module_33(output_module_32)
        output_module_32 = self.module_34(output_module_32)
        output_module_32 = self.module_35(output_module_32)
        output_module_36 = self.module_36(input=output_module_31, other=output_module_32, alpha=1)
        output_module_37 = self.module_37(output_module_36)
        output_module_37 = self.module_38(output_module_37)
        output_module_37 = self.module_39(output_module_37)
        output_module_37 = self.module_40(output_module_37)
        output_module_41 = self.module_41(input=output_module_36, other=output_module_37, alpha=1)
        output_module_42 = self.module_42(output_module_41)
        output_module_42 = self.module_43(output_module_42)
        output_module_42 = self.module_44(output_module_42)
        output_module_42 = self.module_45(output_module_42)
        output_module_46 = self.module_46(input=output_module_41, other=output_module_42, alpha=1)
        output_module_47 = self.module_47(output_module_17)
        output_module_47 = self.module_48(output_module_47)
        output_module_46 = self.module_49(dim=1, tensors=[output_module_46,output_module_47])
        output_module_46 = self.module_50(output_module_46)
        output_module_46 = self.module_51(output_module_46)
        output_module_52 = self.module_52(output_module_46)
        output_module_52 = self.module_53(output_module_52)
        output_module_54 = self.module_54(output_module_52)
        output_module_54 = self.module_55(output_module_54)
        output_module_56 = self.module_56(output_module_54)
        output_module_56 = self.module_57(output_module_56)
        output_module_56 = self.module_58(output_module_56)
        output_module_56 = self.module_59(output_module_56)
        output_module_60 = self.module_60(input=output_module_54, other=output_module_56, alpha=1)
        output_module_61 = self.module_61(output_module_60)
        output_module_61 = self.module_62(output_module_61)
        output_module_61 = self.module_63(output_module_61)
        output_module_61 = self.module_64(output_module_61)
        output_module_65 = self.module_65(input=output_module_60, other=output_module_61, alpha=1)
        output_module_66 = self.module_66(output_module_65)
        output_module_66 = self.module_67(output_module_66)
        output_module_66 = self.module_68(output_module_66)
        output_module_66 = self.module_69(output_module_66)
        output_module_70 = self.module_70(input=output_module_65, other=output_module_66, alpha=1)
        output_module_71 = self.module_71(output_module_70)
        output_module_71 = self.module_72(output_module_71)
        output_module_71 = self.module_73(output_module_71)
        output_module_71 = self.module_74(output_module_71)
        output_module_75 = self.module_75(input=output_module_70, other=output_module_71, alpha=1)
        output_module_76 = self.module_76(output_module_75)
        output_module_76 = self.module_77(output_module_76)
        output_module_76 = self.module_78(output_module_76)
        output_module_76 = self.module_79(output_module_76)
        output_module_80 = self.module_80(input=output_module_75, other=output_module_76, alpha=1)
        output_module_81 = self.module_81(output_module_80)
        output_module_81 = self.module_82(output_module_81)
        output_module_81 = self.module_83(output_module_81)
        output_module_81 = self.module_84(output_module_81)
        output_module_85 = self.module_85(input=output_module_80, other=output_module_81, alpha=1)
        output_module_86 = self.module_86(output_module_52)
        output_module_86 = self.module_87(output_module_86)
        output_module_85 = self.module_88(dim=1, tensors=[output_module_85,output_module_86])
        output_module_85 = self.module_89(output_module_85)
        output_module_85 = self.module_90(output_module_85)
        output_module_91 = self.module_91(output_module_85)
        output_module_91 = self.module_92(output_module_91)
        output_module_93 = self.module_93(output_module_91)
        output_module_93 = self.module_94(output_module_93)
        output_module_95 = self.module_95(output_module_93)
        output_module_95 = self.module_96(output_module_95)
        output_module_95 = self.module_97(output_module_95)
        output_module_95 = self.module_98(output_module_95)
        output_module_99 = self.module_99(input=output_module_93, other=output_module_95, alpha=1)
        output_module_100 = self.module_100(output_module_99)
        output_module_100 = self.module_101(output_module_100)
        output_module_100 = self.module_102(output_module_100)
        output_module_100 = self.module_103(output_module_100)
        output_module_104 = self.module_104(input=output_module_99, other=output_module_100, alpha=1)
        output_module_105 = self.module_105(output_module_91)
        output_module_105 = self.module_106(output_module_105)
        output_module_104 = self.module_107(dim=1, tensors=[output_module_104,output_module_105])
        output_module_104 = self.module_108(output_module_104)
        output_module_104 = self.module_109(output_module_104)
        output_module_104 = self.module_110(output_module_104)
        output_module_104 = self.module_111(output_module_104)
        output_module_112 = self.module_112(output_module_104)
        output_module_113 = self.module_113(output_module_112)
        output_module_114 = self.module_114(output_module_113)
        output_module_115 = self.module_115(dim=1, tensors=[output_module_104,output_module_112,output_module_113,output_module_114])
        output_module_115 = self.module_116(output_module_115)
        output_module_115 = self.module_117(output_module_115)
        output_module_115 = self.module_118(output_module_115)
        output_module_115 = self.module_119(output_module_115)
        output_module_120 = self.module_120(input=output_module_115, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_120 = self.module_121(dim=1, tensors=[output_module_120,output_module_85])
        output_module_122 = self.module_122(output_module_120)
        output_module_122 = self.module_123(output_module_122)
        output_module_122 = self.module_124(output_module_122)
        output_module_122 = self.module_125(output_module_122)
        output_module_122 = self.module_126(output_module_122)
        output_module_122 = self.module_127(output_module_122)
        output_module_122 = self.module_128(output_module_122)
        output_module_122 = self.module_129(output_module_122)
        output_module_122 = self.module_130(output_module_122)
        output_module_122 = self.module_131(output_module_122)
        output_module_132 = self.module_132(output_module_120)
        output_module_132 = self.module_133(output_module_132)
        output_module_122 = self.module_134(dim=1, tensors=[output_module_122,output_module_132])
        output_module_122 = self.module_135(output_module_122)
        output_module_122 = self.module_136(output_module_122)
        output_module_122 = self.module_137(output_module_122)
        output_module_122 = self.module_138(output_module_122)
        output_module_139 = self.module_139(input=output_module_122, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_139 = self.module_140(dim=1, tensors=[output_module_139,output_module_46])
        output_module_141 = self.module_141(output_module_139)
        output_module_141 = self.module_142(output_module_141)
        output_module_141 = self.module_143(output_module_141)
        output_module_141 = self.module_144(output_module_141)
        output_module_141 = self.module_145(output_module_141)
        output_module_141 = self.module_146(output_module_141)
        output_module_141 = self.module_147(output_module_141)
        output_module_141 = self.module_148(output_module_141)
        output_module_141 = self.module_149(output_module_141)
        output_module_141 = self.module_150(output_module_141)
        output_module_151 = self.module_151(output_module_139)
        output_module_151 = self.module_152(output_module_151)
        output_module_141 = self.module_153(dim=1, tensors=[output_module_141,output_module_151])
        output_module_141 = self.module_154(output_module_141)
        output_module_141 = self.module_155(output_module_141)
        output_module_156 = self.module_156(output_module_141)
        output_module_156 = self.module_157(output_module_156)
        output_module_156 = self.module_158(dim=1, tensors=[output_module_156,output_module_122])
        output_module_159 = self.module_159(output_module_156)
        output_module_159 = self.module_160(output_module_159)
        output_module_159 = self.module_161(output_module_159)
        output_module_159 = self.module_162(output_module_159)
        output_module_159 = self.module_163(output_module_159)
        output_module_159 = self.module_164(output_module_159)
        output_module_159 = self.module_165(output_module_159)
        output_module_159 = self.module_166(output_module_159)
        output_module_159 = self.module_167(output_module_159)
        output_module_159 = self.module_168(output_module_159)
        output_module_169 = self.module_169(output_module_156)
        output_module_169 = self.module_170(output_module_169)
        output_module_159 = self.module_171(dim=1, tensors=[output_module_159,output_module_169])
        output_module_159 = self.module_172(output_module_159)
        output_module_159 = self.module_173(output_module_159)
        output_module_174 = self.module_174(output_module_159)
        output_module_174 = self.module_175(output_module_174)
        output_module_174 = self.module_176(dim=1, tensors=[output_module_174,output_module_115])
        output_module_177 = self.module_177(output_module_174)
        output_module_177 = self.module_178(output_module_177)
        output_module_177 = self.module_179(output_module_177)
        output_module_177 = self.module_180(output_module_177)
        output_module_177 = self.module_181(output_module_177)
        output_module_177 = self.module_182(output_module_177)
        output_module_177 = self.module_183(output_module_177)
        output_module_177 = self.module_184(output_module_177)
        output_module_177 = self.module_185(output_module_177)
        output_module_177 = self.module_186(output_module_177)
        output_module_187 = self.module_187(output_module_174)
        output_module_187 = self.module_188(output_module_187)
        output_module_177 = self.module_189(dim=1, tensors=[output_module_177,output_module_187])
        output_module_177 = self.module_190(output_module_177)
        output_module_177 = self.module_191(output_module_177)
        output_module_192 = self.module_192(output_module_141)
        output_module_193 = self.module_193(output_module_159)
        output_module_177 = self.module_194(output_module_177)
        output_module_192 = self.module_195(input=output_module_192)
        output_module_193 = self.module_196(input=output_module_193)
        output_module_177 = self.module_197(input=output_module_177)
        return (output_module_192,output_module_193,output_module_177)
