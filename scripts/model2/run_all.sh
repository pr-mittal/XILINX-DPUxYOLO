#!/bin/bash
#yolox
python compile_data.py

./docker_run.sh xilinx/vitis-ai-pytorch-cpu
CONDA_BASE=$(conda info --base)
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
# conda activate vitis-ai-optimizer_pytorch
sudo ln -s  /usr/lib/x86_64-linux-gnu/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so.6
conda activate vitis-ai-pytorch

#getting errors do doing pip requirements
# pip install -r requirements.txt
pip install pycocotools thop tensorboard

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}
export PYTHONPATH=./code

### Train/Eval/QAT
# 1. Evaluation
#   - Execute run_eval.sh.
bash code/run_eval.sh
# 2. Training
bash code/run_train.sh
# 3. Model quantization and xmodel dumping
bash code/run_quant.sh
# 4. QAT(Quantization-Aware-Training), model converting and xmodel dumping
#   - Configure the variables and in `code/run_qat.sh`, read the steps(including QAT, model testing, model converting and xmodel dumping) in the script and run the step you want.
bash code/run_qat.sh

### 
# # compile for target boards
# source compile.sh zcu102 ${BUILD} ${LOG}
# source compile.sh zcu104 ${BUILD} ${LOG}
# source compile.sh u50 ${BUILD} ${LOG}
# source compile.sh vck190 ${BUILD} ${LOG}
source compile.sh kv260 ${BUILD} ${LOG}
python -u target.py --target kv260  -d ${BUILD} 2>&1 | tee ${LOG}/target_kv260.log