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

# -*- coding: utf-8 -*-
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.use_darknet_cfg = True
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg')

Cfg.batch = 32
Cfg.imgsz = 640
Cfg.width = 640
Cfg.height = 640
Cfg.channels = 3
Cfg.momentum = 0.937
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1
Cfg.weight_decay= 0.0005
Cfg.lr0 = 0.01
Cfg.lrf = 0.1
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.weight_decay=0.0005  # optimizer weight decay 5e-4
Cfg.warmup_epochs=3.0  # warmup epochs (fractions ok)
Cfg.warmup_momentum=0.8  # warmup initial momentum
Cfg.warmup_bias_lr = 0.1  # warmup initial bias lr

Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 80
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_EPOCHS = 300
# Cfg.train_label = os.path.join(_BASE_DIR, 'data', 'train.txt')
# Cfg.val_label = os.path.join(_BASE_DIR, 'data' ,'val.txt')
Cfg.TRAIN_OPTIMIZER = 'adamw'
Cfg.box = 0.05  # box loss gain
Cfg.cls = 0.5  # cls loss gain
Cfg.cls_pw = 1.0  # cls BCELoss positive_weight
Cfg.obj = 1.0  # obj loss gain (scale with pixels)
Cfg.obj_pw = 1.0  # obj BCELoss positive_weight
Cfg.fl_gamma = 0.0  # focal loss gamma (efficientDet default gamma=1.5)
Cfg.mixup = 0.0  # image mixup (probability)


Cfg.checkpoints = os.path.join(_BASE_DIR, 'build/checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'build/log')

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg.keep_checkpoint_max = 10

###
Cfg.anchor_threshold = 4.0
Cfg.scale = 0.5
Cfg.anchor_t = 4.0  # anchor-multiple threshold
#COCO data
Cfg.data_path = 'dataset/' #'/proj/xcdhdstaff1/shangton/coco/'
Cfg.train_image = 'train'
Cfg.val_image = 'val'

Cfg.class_num = 7
Cfg.class_names = ['Motor Vehicle','Non-motorized Vehicle','Pedestrian','Traffic Light-Red Light' ,'Traffic Light-Yellow Light','Traffic Light-Green Light' ,'Traffic Light-Off']  # class names
Cfg.class_map = [1, 2, 3, 4, 5, 6, 7]
Cfg.hsv_h = 0.015  # image HSV-Hue augmentation (fraction)
Cfg.hsv_s = 0.7  # image HSV-Saturation augmentation (fraction)
Cfg.hsv_v = 0.4  # image HSV-Value augmentation (fraction)
Cfg.degrees = 0.0  # image rotation (+/- deg)degrees
Cfg.translate = 0.1  # image translation (+/- fraction)
Cfg.scale = 0.5  # image scale (+/- gain)
Cfg.shear = 0.0  # image shear (+/- deg)
Cfg.rect = False
Cfg.image_weights = False
Cfg.quad = False