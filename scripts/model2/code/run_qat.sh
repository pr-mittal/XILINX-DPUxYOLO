
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

export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1

WEIGHTS=${PWD}/float/yolox.pth
QAT_DIR=${PWD}/qat
BATCH_SIZE=24
CFG=code/exps/default/yolox_s_deploy_qat.py

# Step1: QAT
python code/tools/train.py -f ${CFG} -d ${GPU_NUM} -b ${BATCH_SIZE} -o

# Step2: Eval accuracy after QAT
QAT_WEIGHTS=qat/qat.pth  # assign the path to your QAT weights
python code/tools/eval.py -f ${CFG} -c ${QAT_WEIGHTS} -b ${BATCH_SIZE} -d ${GPU_NUM} --conf 0.001

# Step3: Convert the QAT weights to float.pth + quant_info.json
CVT_DIR=qat/convert_qat_results
python code/tools/convert_qat.py -f ${CFG} -c ${QAT_WEIGHTS} --cvt_dir ${CVT_DIR}


############################# test the converted results #############################

Q_DIR=${CVT_DIR}
CVT_WEIGHTS=${CVT_DIR}/converted_qat.pth
CFG=code/exps/default/yolox_s_deploy.py

export W_QUANT=1

# Step4: Test the converted QAT results (float.pth + quant_info.json) in the similar way with the quantization process, the only difference is we set '--nndct_equalization=False' and '--nndct_param_corr=False'
MODE='test'
python code/tools/quant.py -f ${CFG} -c ${CVT_WEIGHTS} -b ${BATCH_SIZE} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} --nndct_equalization=False --nndct_param_corr=False

# Step5: Dump xmodel for deployment with '--is_dump'
MODE='test'
python code/tools/quant.py -f ${CFG} -c ${CVT_WEIGHTS} -b ${BATCH_SIZE} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} --nndct_equalization=False --nndct_param_corr=False --is_dump

