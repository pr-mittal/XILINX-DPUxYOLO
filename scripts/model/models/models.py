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

import torch
from torch import nn
from models.yolo_base_30 import Model as base_model_30
from models.yolo_base_0 import Model as base_model_0
from models.yolo_base_50 import Model as base_model_50
from pytorch_nndct.nn import QuantStub, DeQuantStub

class Post_Process(nn.Module):
    def __init__(self, anchors=(),is_train = False):  # detection layer
        super().__init__()
        self.nc = 7  # number of classes
        self.no = self.nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.stride = None
        self.is_train = is_train

    def method(self, x):
        x = list(x)
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,(7+5)*3,20,20) to x(bs,3,20,20,12)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, x, y, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(y).to(d), torch.arange(x).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, y, x, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, y, x, 2)).float()
        return grid, anchor_grid


class ofa_yolo_30(nn.Module):
    def __init__(self,anchors=(),anchors_weight=None):
        super().__init__()
        # ch = 3
        self.model = base_model_30()
        self.m = Post_Process(anchors=anchors)
        self.m.stride = self.model.stride
        if anchors_weight is not None:
            self.m.anchors = anchors_weight
        self.stride = self.m.stride
        self.names = self.model.names
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x,quant = False):
        x = self.quant(x)
        x = self.model(x)
        if not quant:
            x = self.m.method(x)
        x = self.dequant(x)
        return x
    def freeze(self):
        for name,param in self.model.named_parameters():
            if name not in ['module_240.weight', 'module_240.bias', 'module_241.weight', 'module_241.bias', 'module_242.weight', 'module_242.bias']:
                # print(name)
                param.requires_grad=False
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad_()

class ofa_yolo_0(nn.Module):
    def __init__(self,anchors=(),anchors_weight=None):
        super().__init__()
        # ch = 3
        self.model = base_model_0()
        self.m = Post_Process(anchors=anchors)
        self.m.stride = self.model.stride
        if anchors_weight is not None:
            self.m.anchors = anchors_weight
        self.stride = self.m.stride
        self.names = self.model.names
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x,quant = False):
        x = self.quant(x)
        x = self.model(x)
        if not quant:
            x = self.m.method(x)
        x = self.dequant(x)
        return x
    def freeze(self):
        for name,param in self.model.named_parameters():
            if name not in ['module_240.weight', 'module_240.bias', 'module_241.weight', 'module_241.bias', 'module_242.weight', 'module_242.bias']:
                # print(name)
                param.requires_grad=False
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad_()
class ofa_yolo_50(nn.Module):
    def __init__(self,anchors=(),anchors_weight=None):
        super().__init__()
        # ch = 3
        self.model = base_model_50()
        self.m = Post_Process(anchors=anchors)
        self.m.stride = self.model.stride
        if anchors_weight is not None:
            self.m.anchors = anchors_weight
        self.stride = self.m.stride
        self.names = self.model.names
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x,quant = False):
        x = self.quant(x)
        x = self.model(x)
        if not quant:
            x = self.m.method(x)
        x = self.dequant(x)
        return x
    def freeze(self):
        for name,param in self.model.named_parameters():
            if name not in ['module_240.weight', 'module_240.bias', 'module_241.weight', 'module_241.bias', 'module_242.weight', 'module_242.bias']:
                # print(name)
                param.requires_grad=False
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad_()


