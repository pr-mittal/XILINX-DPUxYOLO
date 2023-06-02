#!/bin/bash
#yolo
python compile_data.py
git clone https://github.com/ultralytics/yolov5  # clone
git clone https://github.com/AlessandroMondin/YOLOV5m.git
git clone https://github.com/eriklindernoren/PyTorch-YOLOv3

# mv yolov3/pytorchyolo/train.py yolov3/train.py
# mv yolov3/pytorchyolo/test.py yolov3/test.py 
# mv yolov3/pytorchyolo/detect.py yolov3/detect.py 

# python yolov3_test.py --weights weights/yolov3.weights
# python yolov3_detect.py --images yolov3/data/samples/
# python yolov3_train.py --data dataset/dacsdc.data  --pretrained_weights yolov3/weights/darknet53.conv.74

# import cv2
# from pytorchyolo import detect, models

# # Load the YOLO model
# model = models.load_model(
#   "<PATH_TO_YOUR_CONFIG_FOLDER>/yolov3.cfg",
#   "<PATH_TO_YOUR_WEIGHTS_FOLDER>/yolov3.weights")

# # Load the image as a numpy array
# img = cv2.imread("<PATH_TO_YOUR_IMAGE>")

# # Convert OpenCV bgr to rgb
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Runs the YOLO model on the image
# boxes = detect.detect_image(model, img)

# print(boxes)
# # Output will be a numpy array in the following format:
# # [[x1, y1, x2, y2, confidence, class]]

python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt  # install

cd yolov5
cp ../dataset/dacsdc.yaml data/dacsdc.yaml
# run training
# python -u train.py -d ${BUILD} 2>&1 | tee ${LOG}/train.log
python train.py --data dacsdc.yaml --epochs 300 --weights yolov5n.pt --cfg yolov5n.yaml  --batch-size 128
# yolov5n.yaml  --batch-size 128
# yolov5s                    64
# yolov5m                    40
# yolov5l                    24
# yolov5x                    16

# python detect.py --weights yolov5s.pt --source 0
# 0                               # webcam
# img.jpg                         # image
# vid.mp4                         # video
# screen                          # screenshot
# path/                           # directory
# list.txt                        # list of images
# list.streams                    # list of streams
# 'path/*.jpg'                    # glob
# 'https://youtu.be/Zgi9g1ksQHc'  # YouTube
# 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
cd ..
python convert_model.py

./docker_run.sh xilinx/vitis-ai-opt-pytorch-gpu
CONDA_BASE=$(conda info --base)
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
# conda activate vitis-ai-optimizer_pytorch
sudo ln -s  /usr/lib/x86_64-linux-gnu/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so.6
conda activate vitis-ai-pytorch
pip install -r requirements.txt
# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

# quantize & export quantized model
python -u yolov3_quantize.py --weights yolov3/weights/yolov3.weights -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
python -u yolov3_quantize.py --weights yolov3/weights/yolov3.weights -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log


# compile for target boards
source compile.sh zcu102 ${BUILD} ${LOG}
source compile.sh zcu104 ${BUILD} ${LOG}
source compile.sh u50 ${BUILD} ${LOG}
source compile.sh vck190 ${BUILD} ${LOG}
source compile.sh kv260 ${BUILD} ${LOG}

# make target folders
python -u target.py --target zcu102 -d ${BUILD} 2>&1 | tee ${LOG}/target_zcu102.log
python -u target.py --target zcu104 -d ${BUILD} 2>&1 | tee ${LOG}/target_zcu104.log
python -u target.py --target vck190 -d ${BUILD} 2>&1 | tee ${LOG}/target_vck190.log
python -u target.py --target u50    -d ${BUILD} 2>&1 | tee ${LOG}/target_u50.log
python -u target.py --target kv260  -d ${BUILD} 2>&1 | tee ${LOG}/target_kv260.log

