#!/usr/bin/env bash
python compile_data.py
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # install
# python train.py --img 320 --epochs 3 --data dacsdc.yaml --weights yolov5s.pt
python train.py --img 128 --epochs 3 --data dacsdc.yaml --weights yolov5s.pt
