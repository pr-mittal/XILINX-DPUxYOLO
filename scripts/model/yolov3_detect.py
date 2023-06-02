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

from models.models import ofa_yolo_30,ofa_yolo_50,ofa_yolo_0
from PIL import Image
import random
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
import torch.nn as nn
from functools import partial
from tqdm import tqdm
from cfg import Cfg
from easydict import EasyDict as edict
from utils.datasets import Yolo_dataset
from utils.utils import nms,ap_per_class
from utils.torch_utils import box_ious
from utils.utils import xywh2xyxy,xyxy2xywh,scale_coords
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})
def evaluate(model=None,
        cfg=None,
        device=None,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        nndct_quant=False,
        dump_xmodel=False,  # train, val, test, speed or study
        task='val',
        all_result=False,
        save_coco=True,
        train_val = False,
        deploy_dir = ''
        ):
    
    batch_size = cfg.batchsize
    if nndct_quant and not dump_xmodel:
        batch_size = 32
    imgsz = cfg.imgsz
    conf_thres = cfg.conf_thres
    iou_thres = cfg.iou_thres
    nw = cfg.num_worker  # number of workers

    detect_image = cfg.images

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if not nndct_quant:
        if half:
            model.half()
    # model = model.cuda()

    # Configure
    model.eval()
    class_num = cfg.class_num
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once pad = 0.0 if task == 'speed' else 0.5

    pad = 0.0
    dataset = Yolo_dataset(detect_image, imgsz, batch_size,  pad=pad, rect=not nndct_quant)

    dataloader = torch.utils.data.DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        pin_memory=True,
                        collate_fn=dataset.collate_fn)


    seen = 0
    class_map = cfg.class_map
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    
    total = len(dataloader)
    if dump_xmodel:
        total = 1
    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s, total=total)):
        img = img.to(device, non_blocking=True)
        targets = targets.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        bs, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            t1 = time.time()
            out = model(img)
            inf_out, train_out = out[0], out[1]
            t2 = time.time()

        dt[0] += t2 - t1


        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = []  # for autolabelling
        output = nms(inf_out, conf_thres, iou_thres)
        dt[1] += time.time() - t2
        dt[2] += time.time() - t1
        # Store image and detections
        img_detections.extend(output)
        imgs.extend(paths)
    _draw_and_save_output_images(img_detections, imgs, cfg.imgsz, os.path.join(ROOT,"build/detect"), cfg.class_names)


def _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes):
    """Draws detections in output images and stores them.

    :param img_detections: List of detections
    :type img_detections: [Tensor]
    :param imgs: List of paths to image files
    :type imgs: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
     # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)

    # Iterate through images and save plot of detections
    for (image_path, detections) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        _draw_and_save_output_image(
            image_path, detections, img_size, output_path, classes)


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    # print(image_path)
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=f"{classes[int(cls_pred)]}: {conf:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


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
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    parser.add_argument('--save-dir', default='./build/val', help='save to save_dir')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--nndct_quant', action='store_true', help='Train nndct QAT model')
    parser.add_argument('--qat_group', action='store_true', help='param groups')
    parser.add_argument('--ratio', default=30, type=int, help='pruning ratio')
    parser.add_argument('--num_worker', default=8, help='number of workers')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument("-i", "--images", type=str, default="dataset", help="Path to directory with images to inference")
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)

def main():
    logging = init_logger(log_dir='build/log_test')
    cfg = get_args(**Cfg)
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

    # model = Model(anchors, anchors_weight)
    del checkpoint['anchors']
    model.model.load_state_dict(checkpoint, strict=True)
    print('load successfully')

    model = model.to(device)
    evaluate(model,cfg=cfg,device=device)

if __name__ == "__main__":
    main()