#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import random
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from terminaltables import AsciiTable
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from yolov3.pytorchyolo.models import load_model
from yolov3.pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info,ap_per_class,get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy
from yolov3.pytorchyolo.utils.datasets import ImageFolder
from yolov3.pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from utils.datasets import ListDataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from models.models import ofa_yolo_30,ofa_yolo_50,ofa_yolo_0

import json
import os
import sys
from pathlib import Path
import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime
import numpy as np
import torch
from cfg import Cfg
from easydict import EasyDict as edict
from test import evaluate
from pytorch_nndct.apis import torch_quantizer
import pytorch_nndct as py_nndct
from nndct_shared.utils import NndctOption
from nndct_shared.base import key_names, NNDCT_KEYS, NNDCT_DEBUG_LVL, GLOBAL_MAP, NNDCT_OP
import nndct_shared.quantization as nndct_quant
from pytorch_nndct.quantization import torchquantizer
from functools import partial

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    mode: 'a', append; 'w', cover.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging
def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                         help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0',
                        help='GPU', dest='gpu')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('--save-dir', default='./build/val_quant', help='save to save_dir')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--nndct_quant', action='store_true', help='Train nndct QAT model')
    parser.add_argument('--qat_group', action='store_true', help='param groups')
    parser.add_argument('--ratio', default=30, type=int, help='pruning ratio')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--subset_len', help='The size of dataset for fast finetune',type=int,default=1024)
    parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'], help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
    parser.add_argument('--dump_xmodel', action='store_true', default=False)
    parser.add_argument('--fast_finetune', action='store_true', default=False)
    parser.add_argument('--qt_dir', default='./build/nndct_quant', help='save to quant_info')
    parser.add_argument('--nndct_bitwidth', type=int, default=8, help='save to quant_info')
    parser.add_argument('--log_dir', default='./build/log', help='save to log')
    parser.add_argument('--num_worker', default=4, help='number of workers')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)


def quantization(model,cfg=None,device=None):

    if cfg.quant_mode != 'test' and cfg.dump_xmodel:
        cfg.dump_xmodel = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if cfg.dump_xmodel and cfg.batchsize != 1:
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        cfg.batchsize = 1
    model.eval()
    input_tensor = (torch.zeros(1, 3, cfg.imgsz, cfg.imgsz).to(device).type_as(next(model.parameters())))
    model.forward = partial(model.forward, quant=True)
    output_dir = cfg.qt_dir
    print(f"NNDCT quant dir: {output_dir}")
    quantizer = torch_quantizer(quant_mode=cfg.quant_mode,
                                bitwidth=cfg.nndct_bitwidth,
                                module=model,
                                input_args=input_tensor,
                                output_dir=output_dir)

    quant_model = quantizer.quant_model
    ori_forward = quant_model.forward
    post_method = model.m.method
    def forward(x):
        out = ori_forward(x)
        return post_method(out)
    quant_model.forward = forward
    quant_model.stride = model.stride
    print("========== eval after quantization ==========")
    if cfg.dump_xmodel:
        evaluate(quant_model, cfg=cfg,device=device,half=not cfg.nndct_quant,nndct_quant = cfg.nndct_quant,dump_xmodel=True)
    else:
        evaluate(quant_model, cfg=cfg, device=device, half=not cfg.nndct_quant, nndct_quant=cfg.nndct_quant)
    if cfg.fast_finetune:
        if cfg.quant_mode == 'calib':
            quantizer.fast_finetune(evaluate, (quant_model,cfg,device,not cfg.nndct_quant,cfg.nndct_quant))
    if cfg.quant_mode == 'calib':
        quantizer.export_quant_config()
    if cfg.dump_xmodel and cfg.quant_mode == 'test':
        quantizer.export_xmodel(output_dir=cfg.qt_dir, deploy_check=True)

def main():
    cfg = get_args(**Cfg)
    logging = init_logger(log_dir=cfg.log_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if cfg.nndct_quant:
        os.environ["W_QUANT"] = "1"
    # # model
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    pretrained_ofa_model = cfg.pretrained 
    with open(pretrained_ofa_model, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
    anchors_weight = checkpoint['anchors']
    if cfg.ratio == 30:
        model = ofa_yolo_30(anchors, anchors_weight)
    elif cfg.ratio == 0:
        model = ofa_yolo_0(anchors, anchors_weight)
    elif cfg.ratio == 50:
        model = ofa_yolo_50(anchors, anchors_weight)
    del checkpoint['anchors']
    model.model.load_state_dict(checkpoint, strict=True)
    print('load successfully')
    model = model.cuda()

    # calibration or evaluation
    quantization(model,cfg,device)


if __name__ == '__main__':
    main()