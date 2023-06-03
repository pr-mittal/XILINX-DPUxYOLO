
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

echo "Conducting quantization..."

export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1

CFG=code/exps/default/yolox_s_deploy.py
WEIGHTS=${PWD}/float/yolox.pth # assign the path to your float weights
Q_DIR=${PWD}/quantized

export W_QUANT=1
BATCH=1
Q_DIR='quantized'

# Step1: Calibration
MODE='calib'
python code/tools/quant.py -f ${CFG} -c ${WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR}

# Step2: Test accuracy for quantized model
MODE='test'
python code/tools/quant.py -f ${CFG} -c ${WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR}

# Step3: Dump xmodel for deployment
MODE='test'
python code/tools/quant.py -f ${CFG} -c ${WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} --is_dump


##################################### fast_finetune #####################################

# BATCH=32
# Q_DIR='quantized_fft'

# MODE='calib'
# python code/tools/quant.py -f ${CFG} -c ${WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} --fast_finetune

# MODE='test'
# python code/tools/quant.py -f ${CFG} -c ${WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} --fast_finetune
# python code/tools/quant.py -f ${CFG} -c ${WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} --fast_finetune --is_dump
