#!/bin/bash
#yolo
python compile_data.py

./docker_run.sh xilinx/vitis-ai-pytorch-cpu
CONDA_BASE=$(conda info --base)
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
# conda activate vitis-ai-optimizer_pytorch
sudo ln -s  /usr/lib/x86_64-linux-gnu/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so.6
conda activate vitis-ai-pytorch

#getting errors do doing pip requirements
# pip install -r requirements.txt
pip install pycocotools

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

./run_qat.sh
./run_quant.sh
./run_qat_test.sh
./run_test.sh
# # compile for target boards
# source compile.sh zcu102 ${BUILD} ${LOG}
# source compile.sh zcu104 ${BUILD} ${LOG}
# source compile.sh u50 ${BUILD} ${LOG}
# source compile.sh vck190 ${BUILD} ${LOG}
source compile.sh kv260 ${BUILD} ${LOG}
python -u target.py --target kv260  -d ${BUILD} 2>&1 | tee ${LOG}/target_kv260.log