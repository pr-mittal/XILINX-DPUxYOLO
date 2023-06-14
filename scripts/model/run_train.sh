#!/bin/bash
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

CUDA_VISIBLE_DEVICES='0' python yolov3_train.py --batch-size 16 --epoch 60 --gpu 1 --log_dir ./build/log/log_yolov3_train_0 --save-dir ./build/yolov3_train_0 --save-period 1 -pretrained ./build/float_model/yolo_base_0.pth 

#CUDA_VISIBLE_DEVICES='0' python yolov3_train.py --batch-size 16 --epoch 60 --gpu 1 --log_dir ./build/log/log_yolov3_train_30 --save-dir ./build/yolov3_train_30 --save-period 1 -pretrained /group/modelzoo/internal-cooperation-models/pytorch/ofa_yolo/float/yolo_base_30.pth 

#CUDA_VISIBLE_DEVICES='0' python yolov3_train.py --batch-size 16 --epoch 60 --gpu 1 --log_dir ./build/log/log_yolov3_train_50 --save-dir ./build/yolov3_train_50 --save-period 1 -pretrained /group/modelzoo/internal-cooperation-models/pytorch/ofa_yolo/float/yolo_base_50.pth 