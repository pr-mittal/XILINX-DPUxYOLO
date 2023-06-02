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

#PTQ
CUDA_VISIBLE_DEVICES='0' python yolov3_quant.py --batch-size 32 --gpu 6 -pretrained build/float_model/yolo_base_0.pth --ratio 0 --log_dir ./build/log_yolov3_quant_0 --qt_dir ./build/nndct_yolov3_quant_0 --nndct_quant --quant_mode calib

#CUDA_VISIBLE_DEVICES='0' python yolov3_quant.py --batch-size 32 --gpu 6 -pretrained build/float_model/yolo_base_30.pth --ratio 30 --log_dir ./build/log_yolov3_quant_30 --qt_dir ./build/nndct_yolov3_quant_30 --nndct_quant --quant_mode calib

#CUDA_VISIBLE_DEVICES='0' python yolov3_quant.py --batch-size 32 --gpu 6 -pretrained build/float_model/yolo_base_50.pth --ratio 50 --log_dir ./build/log_yolov3_quant_50 --qt_dir ./build/nndct_yolov3_quant_50 --nndct_quant --quant_mode calib

#dump xmodel
#CUDA_VISIBLE_DEVICES='0' python yolov3_quant.py --batch-size 1 --gpu 6 -pretrained /group/modelzoo/internal-cooperation-models/pytorch/ofa_yolo/float/yolo_base_0.pth --ratio 0 --log_dir ./build/log_yolov3_quant_0 --qt_dir ./build/nndct_yolov3_quant_0 --nndct_quant --quant_mode test --dump_xmodel --nndct_equalization=False --nndct_param_corr=False

#CUDA_VISIBLE_DEVICES='0' python yolov3_quant.py --batch-size 1 --gpu 6 -pretrained /group/modelzoo/internal-cooperation-models/pytorch/ofa_yolo/float/yolo_base_30.pth --ratio 30 --log_dir ./build/log_yolov3_quant_30 --qt_dir ./build/nndct_yolov3_quant_30 --nndct_quant --quant_mode test --dump_xmodel --nndct_equalization=False --nndct_param_corr=False

#CUDA_VISIBLE_DEVICES='0' python yolov3_quant.py --batch-size 1 --gpu 6 -pretrained /group/modelzoo/internal-cooperation-models/pytorch/ofa_yolo/float/yolo_base_50.pth --ratio 50 --log_dir ./build/log_yolov3_quant_50 --qt_dir ./build/nndct_yolov3_quant_50 --nndct_quant --quant_mode test --dump_xmodel --nndct_equalization=False --nndct_param_corr=False
