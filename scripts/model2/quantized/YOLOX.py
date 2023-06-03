# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class YOLOX(torch.nn.Module):
    def __init__(self):
        super(YOLOX, self).__init__()
        self.module_0 = py_nndct.nn.Input() #YOLOX::input_0
        self.module_1 = py_nndct.nn.quant_input() #YOLOX::YOLOX/QuantStub[quant_in]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/FocusDeploy[stem]/BaseConv[conv]/Conv2d[conv]/input.2
        self.module_4 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/FocusDeploy[stem]/BaseConv[conv]/LeakyReLU[act]/input.4
        self.module_5 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/BaseConv[0]/Conv2d[conv]/input.5
        self.module_7 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/BaseConv[0]/LeakyReLU[act]/input.7
        self.module_8 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv1]/Conv2d[conv]/input.8
        self.module_10 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv1]/LeakyReLU[act]/input.12
        self.module_11 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv2]/Conv2d[conv]/input.10
        self.module_13 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv2]/LeakyReLU[act]/15894
        self.module_14 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.13
        self.module_16 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/LeakyReLU[act]/input.15
        self.module_17 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/Conv2d[conv]/input.16
        self.module_19 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/LeakyReLU[act]/15948
        self.module_20 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/Add[sc_add]/15950
        self.module_21 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Cat[cat]/input.18
        self.module_22 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv3]/Conv2d[conv]/input.19
        self.module_24 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv3]/LeakyReLU[act]/input.21
        self.module_25 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/BaseConv[0]/Conv2d[conv]/input.22
        self.module_27 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/BaseConv[0]/LeakyReLU[act]/input.24
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv1]/Conv2d[conv]/input.25
        self.module_30 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv1]/LeakyReLU[act]/input.29
        self.module_31 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv2]/Conv2d[conv]/input.27
        self.module_33 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv2]/LeakyReLU[act]/16061
        self.module_34 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.30
        self.module_36 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/LeakyReLU[act]/input.32
        self.module_37 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/Conv2d[conv]/input.33
        self.module_39 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/LeakyReLU[act]/16115
        self.module_40 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/Add[sc_add]/input.35
        self.module_41 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv1]/Conv2d[conv]/input.36
        self.module_43 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv1]/LeakyReLU[act]/input.38
        self.module_44 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv2]/Conv2d[conv]/input.39
        self.module_46 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv2]/LeakyReLU[act]/16171
        self.module_47 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/Add[sc_add]/input.41
        self.module_48 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv1]/Conv2d[conv]/input.42
        self.module_50 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv1]/LeakyReLU[act]/input.44
        self.module_51 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv2]/Conv2d[conv]/input.45
        self.module_53 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv2]/LeakyReLU[act]/16227
        self.module_54 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/Add[sc_add]/16229
        self.module_55 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Cat[cat]/input.47
        self.module_56 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv3]/Conv2d[conv]/input.48
        self.module_58 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv3]/LeakyReLU[act]/input.50
        self.module_59 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/BaseConv[0]/Conv2d[conv]/input.51
        self.module_61 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/BaseConv[0]/LeakyReLU[act]/input.53
        self.module_62 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv1]/Conv2d[conv]/input.54
        self.module_64 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv1]/LeakyReLU[act]/input.58
        self.module_65 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv2]/Conv2d[conv]/input.56
        self.module_67 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv2]/LeakyReLU[act]/16340
        self.module_68 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.59
        self.module_70 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/LeakyReLU[act]/input.61
        self.module_71 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/Conv2d[conv]/input.62
        self.module_73 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/LeakyReLU[act]/16394
        self.module_74 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/Add[sc_add]/input.64
        self.module_75 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv1]/Conv2d[conv]/input.65
        self.module_77 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv1]/LeakyReLU[act]/input.67
        self.module_78 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv2]/Conv2d[conv]/input.68
        self.module_80 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv2]/LeakyReLU[act]/16450
        self.module_81 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/Add[sc_add]/input.70
        self.module_82 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv1]/Conv2d[conv]/input.71
        self.module_84 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv1]/LeakyReLU[act]/input.73
        self.module_85 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv2]/Conv2d[conv]/input.74
        self.module_87 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv2]/LeakyReLU[act]/16506
        self.module_88 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/Add[sc_add]/16508
        self.module_89 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Cat[cat]/input.76
        self.module_90 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv3]/Conv2d[conv]/input.77
        self.module_92 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv3]/LeakyReLU[act]/input.79
        self.module_93 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/BaseConv[0]/Conv2d[conv]/input.80
        self.module_95 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/BaseConv[0]/LeakyReLU[act]/input.82
        self.module_96 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/BaseConv[conv1]/Conv2d[conv]/input.83
        self.module_98 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/BaseConv[conv1]/LeakyReLU[act]/16592
        self.module_99 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], ceil_mode=False) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/MaxPool2d[m]/ModuleList[0]/16606
        self.module_100 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/MaxPool2d[m]/ModuleList[1]/16620
        self.module_101 = py_nndct.nn.MaxPool2d(kernel_size=[7, 7], stride=[1, 1], padding=[3, 3], dilation=[1, 1], ceil_mode=False) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/MaxPool2d[m]/ModuleList[2]/16634
        self.module_102 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/Cat[cat]/input.85
        self.module_103 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/BaseConv[conv2]/Conv2d[conv]/input.86
        self.module_105 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/BaseConv[conv2]/LeakyReLU[act]/input.88
        self.module_106 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv1]/Conv2d[conv]/input.89
        self.module_108 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv1]/LeakyReLU[act]/input.93
        self.module_109 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv2]/Conv2d[conv]/input.91
        self.module_111 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv2]/LeakyReLU[act]/16718
        self.module_112 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.94
        self.module_114 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/LeakyReLU[act]/input.96
        self.module_115 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/Conv2d[conv]/input.97
        self.module_117 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/LeakyReLU[act]/16772
        self.module_118 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Cat[cat]/input.99
        self.module_119 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv3]/Conv2d[conv]/input.100
        self.module_121 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv3]/LeakyReLU[act]/input.102
        self.module_122 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[lateral_conv0]/Conv2d[conv]/input.103
        self.module_124 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[lateral_conv0]/LeakyReLU[act]/input.105
        self.module_125 = py_nndct.nn.Interpolate() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Upsample[upsample]/16838
        self.module_126 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Cat[cat_f0]/input.106
        self.module_127 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv1]/Conv2d[conv]/input.107
        self.module_129 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv1]/LeakyReLU[act]/input.111
        self.module_130 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv2]/Conv2d[conv]/input.109
        self.module_132 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv2]/LeakyReLU[act]/16895
        self.module_133 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.112
        self.module_135 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/LeakyReLU[act]/input.114
        self.module_136 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/Conv2d[conv]/input.115
        self.module_138 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/LeakyReLU[act]/16949
        self.module_139 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Cat[cat]/input.117
        self.module_140 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv3]/Conv2d[conv]/input.118
        self.module_142 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv3]/LeakyReLU[act]/input.120
        self.module_143 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[reduce_conv1]/Conv2d[conv]/input.121
        self.module_145 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[reduce_conv1]/LeakyReLU[act]/input.123
        self.module_146 = py_nndct.nn.Interpolate() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Upsample[upsample]/17011
        self.module_147 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Cat[cat_f1]/input.124
        self.module_148 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv1]/Conv2d[conv]/input.125
        self.module_150 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv1]/LeakyReLU[act]/input.129
        self.module_151 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv2]/Conv2d[conv]/input.127
        self.module_153 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv2]/LeakyReLU[act]/17068
        self.module_154 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.130
        self.module_156 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/LeakyReLU[act]/input.132
        self.module_157 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/Conv2d[conv]/input.133
        self.module_159 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/LeakyReLU[act]/17122
        self.module_160 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Cat[cat]/input.135
        self.module_161 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv3]/Conv2d[conv]/input.136
        self.module_163 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv3]/LeakyReLU[act]/input.138
        self.module_164 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[bu_conv2]/Conv2d[conv]/input.139
        self.module_166 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[bu_conv2]/LeakyReLU[act]/17179
        self.module_167 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Cat[cat_p1]/input.141
        self.module_168 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv1]/Conv2d[conv]/input.142
        self.module_170 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv1]/LeakyReLU[act]/input.146
        self.module_171 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv2]/Conv2d[conv]/input.144
        self.module_173 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv2]/LeakyReLU[act]/17236
        self.module_174 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.147
        self.module_176 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/LeakyReLU[act]/input.149
        self.module_177 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/Conv2d[conv]/input.150
        self.module_179 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/LeakyReLU[act]/17290
        self.module_180 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Cat[cat]/input.152
        self.module_181 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv3]/Conv2d[conv]/input.153
        self.module_183 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv3]/LeakyReLU[act]/input.155
        self.module_184 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[bu_conv1]/Conv2d[conv]/input.156
        self.module_186 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[bu_conv1]/LeakyReLU[act]/17347
        self.module_187 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Cat[cat_p0]/input.158
        self.module_188 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv1]/Conv2d[conv]/input.159
        self.module_190 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv1]/LeakyReLU[act]/input.163
        self.module_191 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv2]/Conv2d[conv]/input.161
        self.module_193 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv2]/LeakyReLU[act]/17404
        self.module_194 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.164
        self.module_196 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/LeakyReLU[act]/input.166
        self.module_197 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/Conv2d[conv]/input.167
        self.module_199 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv2]/LeakyReLU[act]/17458
        self.module_200 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Cat[cat]/input.169
        self.module_201 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv3]/Conv2d[conv]/input.170
        self.module_203 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv3]/LeakyReLU[act]/input.202
        self.module_204 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[0]/Conv2d[conv]/input.172
        self.module_206 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[0]/LeakyReLU[act]/input.174
        self.module_207 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/BaseConv[0]/Conv2d[conv]/input.175
        self.module_209 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/BaseConv[0]/LeakyReLU[act]/input.177
        self.module_210 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/BaseConv[1]/Conv2d[conv]/input.178
        self.module_212 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/BaseConv[1]/LeakyReLU[act]/input.180
        self.module_213 = py_nndct.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[cls_preds]/ModuleList[0]/17592
        self.module_214 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/BaseConv[0]/Conv2d[conv]/input.181
        self.module_216 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/BaseConv[0]/LeakyReLU[act]/input.183
        self.module_217 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/BaseConv[1]/Conv2d[conv]/input.184
        self.module_219 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/BaseConv[1]/LeakyReLU[act]/input.186
        self.module_220 = py_nndct.nn.Conv2d(in_channels=128, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[reg_preds]/ModuleList[0]/17665
        self.module_221 = py_nndct.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[obj_preds]/ModuleList[0]/17684
        self.module_222 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOXHead[head]/Cat[cat_list]/ModuleList[0]/17687
        self.module_223 = py_nndct.nn.dequant_output() #YOLOX::YOLOX/YOLOXHead[head]/DeQuantStub[quant_outs]/ModuleList[0]/17688
        self.module_224 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[1]/Conv2d[conv]/input.187
        self.module_226 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[1]/LeakyReLU[act]/input.189
        self.module_227 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/BaseConv[0]/Conv2d[conv]/input.190
        self.module_229 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/BaseConv[0]/LeakyReLU[act]/input.192
        self.module_230 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/BaseConv[1]/Conv2d[conv]/input.193
        self.module_232 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/BaseConv[1]/LeakyReLU[act]/input.195
        self.module_233 = py_nndct.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[cls_preds]/ModuleList[1]/17789
        self.module_234 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/BaseConv[0]/Conv2d[conv]/input.196
        self.module_236 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/BaseConv[0]/LeakyReLU[act]/input.198
        self.module_237 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/BaseConv[1]/Conv2d[conv]/input.199
        self.module_239 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/BaseConv[1]/LeakyReLU[act]/input.201
        self.module_240 = py_nndct.nn.Conv2d(in_channels=128, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[reg_preds]/ModuleList[1]/17862
        self.module_241 = py_nndct.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[obj_preds]/ModuleList[1]/17881
        self.module_242 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOXHead[head]/Cat[cat_list]/ModuleList[1]/17884
        self.module_243 = py_nndct.nn.dequant_output() #YOLOX::YOLOX/YOLOXHead[head]/DeQuantStub[quant_outs]/ModuleList[1]/17885
        self.module_244 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[2]/Conv2d[conv]/input.203
        self.module_246 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[2]/LeakyReLU[act]/input.205
        self.module_247 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/BaseConv[0]/Conv2d[conv]/input.206
        self.module_249 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/BaseConv[0]/LeakyReLU[act]/input.208
        self.module_250 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/BaseConv[1]/Conv2d[conv]/input.209
        self.module_252 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/BaseConv[1]/LeakyReLU[act]/input.211
        self.module_253 = py_nndct.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[cls_preds]/ModuleList[2]/17986
        self.module_254 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/BaseConv[0]/Conv2d[conv]/input.212
        self.module_256 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/BaseConv[0]/LeakyReLU[act]/input.214
        self.module_257 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/BaseConv[1]/Conv2d[conv]/input.215
        self.module_259 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/BaseConv[1]/LeakyReLU[act]/input
        self.module_260 = py_nndct.nn.Conv2d(in_channels=128, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[reg_preds]/ModuleList[2]/18059
        self.module_261 = py_nndct.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[obj_preds]/ModuleList[2]/18078
        self.module_262 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOXHead[head]/Cat[cat_list]/ModuleList[2]/18081
        self.module_263 = py_nndct.nn.dequant_output() #YOLOX::YOLOX/YOLOXHead[head]/DeQuantStub[quant_outs]/ModuleList[2]/18082

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_8 = self.module_8(output_module_0)
        output_module_8 = self.module_10(output_module_8)
        output_module_11 = self.module_11(output_module_0)
        output_module_11 = self.module_13(output_module_11)
        output_module_14 = self.module_14(output_module_8)
        output_module_14 = self.module_16(output_module_14)
        output_module_14 = self.module_17(output_module_14)
        output_module_14 = self.module_19(output_module_14)
        output_module_14 = self.module_20(input=output_module_14, other=output_module_8, alpha=1)
        output_module_14 = self.module_21(dim=1, tensors=[output_module_14,output_module_11])
        output_module_14 = self.module_22(output_module_14)
        output_module_14 = self.module_24(output_module_14)
        output_module_14 = self.module_25(output_module_14)
        output_module_14 = self.module_27(output_module_14)
        output_module_28 = self.module_28(output_module_14)
        output_module_28 = self.module_30(output_module_28)
        output_module_31 = self.module_31(output_module_14)
        output_module_31 = self.module_33(output_module_31)
        output_module_34 = self.module_34(output_module_28)
        output_module_34 = self.module_36(output_module_34)
        output_module_34 = self.module_37(output_module_34)
        output_module_34 = self.module_39(output_module_34)
        output_module_34 = self.module_40(input=output_module_34, other=output_module_28, alpha=1)
        output_module_41 = self.module_41(output_module_34)
        output_module_41 = self.module_43(output_module_41)
        output_module_41 = self.module_44(output_module_41)
        output_module_41 = self.module_46(output_module_41)
        output_module_41 = self.module_47(input=output_module_41, other=output_module_34, alpha=1)
        output_module_48 = self.module_48(output_module_41)
        output_module_48 = self.module_50(output_module_48)
        output_module_48 = self.module_51(output_module_48)
        output_module_48 = self.module_53(output_module_48)
        output_module_48 = self.module_54(input=output_module_48, other=output_module_41, alpha=1)
        output_module_48 = self.module_55(dim=1, tensors=[output_module_48,output_module_31])
        output_module_48 = self.module_56(output_module_48)
        output_module_48 = self.module_58(output_module_48)
        output_module_59 = self.module_59(output_module_48)
        output_module_59 = self.module_61(output_module_59)
        output_module_62 = self.module_62(output_module_59)
        output_module_62 = self.module_64(output_module_62)
        output_module_65 = self.module_65(output_module_59)
        output_module_65 = self.module_67(output_module_65)
        output_module_68 = self.module_68(output_module_62)
        output_module_68 = self.module_70(output_module_68)
        output_module_68 = self.module_71(output_module_68)
        output_module_68 = self.module_73(output_module_68)
        output_module_68 = self.module_74(input=output_module_68, other=output_module_62, alpha=1)
        output_module_75 = self.module_75(output_module_68)
        output_module_75 = self.module_77(output_module_75)
        output_module_75 = self.module_78(output_module_75)
        output_module_75 = self.module_80(output_module_75)
        output_module_75 = self.module_81(input=output_module_75, other=output_module_68, alpha=1)
        output_module_82 = self.module_82(output_module_75)
        output_module_82 = self.module_84(output_module_82)
        output_module_82 = self.module_85(output_module_82)
        output_module_82 = self.module_87(output_module_82)
        output_module_82 = self.module_88(input=output_module_82, other=output_module_75, alpha=1)
        output_module_82 = self.module_89(dim=1, tensors=[output_module_82,output_module_65])
        output_module_82 = self.module_90(output_module_82)
        output_module_82 = self.module_92(output_module_82)
        output_module_93 = self.module_93(output_module_82)
        output_module_93 = self.module_95(output_module_93)
        output_module_93 = self.module_96(output_module_93)
        output_module_93 = self.module_98(output_module_93)
        output_module_99 = self.module_99(output_module_93)
        output_module_100 = self.module_100(output_module_93)
        output_module_101 = self.module_101(output_module_93)
        output_module_102 = self.module_102(dim=1, tensors=[output_module_93,output_module_99,output_module_100,output_module_101])
        output_module_102 = self.module_103(output_module_102)
        output_module_102 = self.module_105(output_module_102)
        output_module_106 = self.module_106(output_module_102)
        output_module_106 = self.module_108(output_module_106)
        output_module_109 = self.module_109(output_module_102)
        output_module_109 = self.module_111(output_module_109)
        output_module_106 = self.module_112(output_module_106)
        output_module_106 = self.module_114(output_module_106)
        output_module_106 = self.module_115(output_module_106)
        output_module_106 = self.module_117(output_module_106)
        output_module_106 = self.module_118(dim=1, tensors=[output_module_106,output_module_109])
        output_module_106 = self.module_119(output_module_106)
        output_module_106 = self.module_121(output_module_106)
        output_module_106 = self.module_122(output_module_106)
        output_module_106 = self.module_124(output_module_106)
        output_module_125 = self.module_125(input=output_module_106, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_125 = self.module_126(dim=1, tensors=[output_module_125,output_module_82])
        output_module_127 = self.module_127(output_module_125)
        output_module_127 = self.module_129(output_module_127)
        output_module_130 = self.module_130(output_module_125)
        output_module_130 = self.module_132(output_module_130)
        output_module_127 = self.module_133(output_module_127)
        output_module_127 = self.module_135(output_module_127)
        output_module_127 = self.module_136(output_module_127)
        output_module_127 = self.module_138(output_module_127)
        output_module_127 = self.module_139(dim=1, tensors=[output_module_127,output_module_130])
        output_module_127 = self.module_140(output_module_127)
        output_module_127 = self.module_142(output_module_127)
        output_module_127 = self.module_143(output_module_127)
        output_module_127 = self.module_145(output_module_127)
        output_module_146 = self.module_146(input=output_module_127, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_146 = self.module_147(dim=1, tensors=[output_module_146,output_module_48])
        output_module_148 = self.module_148(output_module_146)
        output_module_148 = self.module_150(output_module_148)
        output_module_151 = self.module_151(output_module_146)
        output_module_151 = self.module_153(output_module_151)
        output_module_148 = self.module_154(output_module_148)
        output_module_148 = self.module_156(output_module_148)
        output_module_148 = self.module_157(output_module_148)
        output_module_148 = self.module_159(output_module_148)
        output_module_148 = self.module_160(dim=1, tensors=[output_module_148,output_module_151])
        output_module_148 = self.module_161(output_module_148)
        output_module_148 = self.module_163(output_module_148)
        output_module_164 = self.module_164(output_module_148)
        output_module_164 = self.module_166(output_module_164)
        output_module_164 = self.module_167(dim=1, tensors=[output_module_164,output_module_127])
        output_module_168 = self.module_168(output_module_164)
        output_module_168 = self.module_170(output_module_168)
        output_module_171 = self.module_171(output_module_164)
        output_module_171 = self.module_173(output_module_171)
        output_module_168 = self.module_174(output_module_168)
        output_module_168 = self.module_176(output_module_168)
        output_module_168 = self.module_177(output_module_168)
        output_module_168 = self.module_179(output_module_168)
        output_module_168 = self.module_180(dim=1, tensors=[output_module_168,output_module_171])
        output_module_168 = self.module_181(output_module_168)
        output_module_168 = self.module_183(output_module_168)
        output_module_184 = self.module_184(output_module_168)
        output_module_184 = self.module_186(output_module_184)
        output_module_184 = self.module_187(dim=1, tensors=[output_module_184,output_module_106])
        output_module_188 = self.module_188(output_module_184)
        output_module_188 = self.module_190(output_module_188)
        output_module_191 = self.module_191(output_module_184)
        output_module_191 = self.module_193(output_module_191)
        output_module_188 = self.module_194(output_module_188)
        output_module_188 = self.module_196(output_module_188)
        output_module_188 = self.module_197(output_module_188)
        output_module_188 = self.module_199(output_module_188)
        output_module_188 = self.module_200(dim=1, tensors=[output_module_188,output_module_191])
        output_module_188 = self.module_201(output_module_188)
        output_module_188 = self.module_203(output_module_188)
        output_module_204 = self.module_204(output_module_148)
        output_module_204 = self.module_206(output_module_204)
        output_module_207 = self.module_207(output_module_204)
        output_module_207 = self.module_209(output_module_207)
        output_module_207 = self.module_210(output_module_207)
        output_module_207 = self.module_212(output_module_207)
        output_module_207 = self.module_213(output_module_207)
        output_module_214 = self.module_214(output_module_204)
        output_module_214 = self.module_216(output_module_214)
        output_module_214 = self.module_217(output_module_214)
        output_module_214 = self.module_219(output_module_214)
        output_module_220 = self.module_220(output_module_214)
        output_module_221 = self.module_221(output_module_214)
        output_module_220 = self.module_222(dim=1, tensors=[output_module_220,output_module_221,output_module_207])
        output_module_220 = self.module_223(input=output_module_220)
        output_module_224 = self.module_224(output_module_168)
        output_module_224 = self.module_226(output_module_224)
        output_module_227 = self.module_227(output_module_224)
        output_module_227 = self.module_229(output_module_227)
        output_module_227 = self.module_230(output_module_227)
        output_module_227 = self.module_232(output_module_227)
        output_module_227 = self.module_233(output_module_227)
        output_module_234 = self.module_234(output_module_224)
        output_module_234 = self.module_236(output_module_234)
        output_module_234 = self.module_237(output_module_234)
        output_module_234 = self.module_239(output_module_234)
        output_module_240 = self.module_240(output_module_234)
        output_module_241 = self.module_241(output_module_234)
        output_module_240 = self.module_242(dim=1, tensors=[output_module_240,output_module_241,output_module_227])
        output_module_240 = self.module_243(input=output_module_240)
        output_module_188 = self.module_244(output_module_188)
        output_module_188 = self.module_246(output_module_188)
        output_module_247 = self.module_247(output_module_188)
        output_module_247 = self.module_249(output_module_247)
        output_module_247 = self.module_250(output_module_247)
        output_module_247 = self.module_252(output_module_247)
        output_module_247 = self.module_253(output_module_247)
        output_module_254 = self.module_254(output_module_188)
        output_module_254 = self.module_256(output_module_254)
        output_module_254 = self.module_257(output_module_254)
        output_module_254 = self.module_259(output_module_254)
        output_module_260 = self.module_260(output_module_254)
        output_module_261 = self.module_261(output_module_254)
        output_module_260 = self.module_262(dim=1, tensors=[output_module_260,output_module_261,output_module_247])
        output_module_260 = self.module_263(input=output_module_260)
        return output_module_220,output_module_240,output_module_260
