# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class CNN(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(CNN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CNN::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[5, 5], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[network]/Conv2d[0]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[network]/ReLU[2]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[5, 5], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[network]/Conv2d[3]/input.9
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[network]/ReLU[5]/input.13
        self.module_5 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[network]/Conv2d[6]/input.15
        self.module_6 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[network]/ReLU[8]/input.19
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=10, kernel_size=[3, 3], stride=[3, 3], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[network]/Conv2d[9]/input
        self.module_8 = py_nndct.nn.Module('nndct_flatten') #CNN::CNN/Sequential[network]/Flatten[11]/508

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(input=output_module_0, start_dim=1, end_dim=-1)
        return output_module_0
