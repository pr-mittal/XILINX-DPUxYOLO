
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

# train
export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1
CFG=code/exps/default/yolox_s_deploy.py
WEIGHTS=float/yolox.pth

# demo
IMG=data/KITTI/training/image_2/000025.png
export W_QUANT=0 # float model
# export W_QUANT=1 # quantized model
python code/tools/demo_kitti.py image -f ${CFG} -c ${WEIGHTS} --path ${IMG} --conf 0.25 --nms 0.45 --tsize 384 1248 --save_result --device gpu
