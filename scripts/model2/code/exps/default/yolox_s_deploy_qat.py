#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import ExpKITTIDeployQat as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.is_qat = True
        self.float_ckpt = 'float/yolox.pth'
        self.calib_dir = 'quantized'
        self.thresh_lr_scale = 10
