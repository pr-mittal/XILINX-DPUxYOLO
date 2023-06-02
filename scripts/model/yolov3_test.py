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


    # Directories
    if not os.path.exists(cfg.save_dir):
        os.mkdir(cfg.save_dir)
    save_dir = cfg.save_dir + '/test'  # increment run
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    val_image = cfg.data_path + cfg.val_image

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

    pad = 0.5
    dataset = Yolo_dataset(val_image, imgsz, batch_size,  pad=pad, rect=not nndct_quant)

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
    if nndct_quant and train_val:
        from pytorch_nndct.apis import torch_quantizer
        import pytorch_nndct as py_nndct
        from nndct_shared.utils import NndctOption
        from nndct_shared.base import key_names, NNDCT_KEYS, NNDCT_DEBUG_LVL, GLOBAL_MAP, NNDCT_OP
        import nndct_shared.quantization as nndct_quant
        from pytorch_nndct.quantization import torchquantizer

        input_tensor = (torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        model.forward = partial(model.forward, quant=True)

        output_dir = deploy_dir
        print(f"NNDCT quant dir: {output_dir}")
        quantizer = torch_quantizer(quant_mode='test',
                                    bitwidth=8,
                                    module=model,
                                    input_args=input_tensor,
                                    output_dir=output_dir if isinstance(output_dir, str) else output_dir.as_posix())
        if (NndctOption.nndct_stat.value > 2):
            def do_quantize(instance, blob, name, node=None, tensor_type='input'):
                # forward quant graph but not quantize parameter and activation
                if NndctOption.nndct_quant_off.value:
                    return blob

                blob_save = None
                if isinstance(blob.values, torch.Tensor):
                    blob_save = blob
                    blob = blob.values.data

                quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
                if blob.device.type != quant_device.type:
                    raise TypeError(
                        "Device of quantizer is {}, device of model and data should match device of quantizer".format(
                            quant_device.type))

                if (NndctOption.nndct_stat.value > 2):
                    quant_data = nndct_quant.QuantizeData(name, blob.cpu().detach().numpy())
                # quantize the tensor
                bnfp = instance.get_bnfp(name, True, tensor_type)
                if (NndctOption.nndct_stat.value > 1):
                    print('---- quant %s tensor: %s with 1/step = %g' % (
                        tensor_type, name, bnfp[1]))
                # hardware cut method
                mth = 4 if instance.lstm else 2
                if tensor_type == 'param':
                    mth = 3

                res = py_nndct.nn.NndctFixNeuron(blob,
                                                 blob,
                                                 maxamp=[bnfp[0], bnfp[1]],
                                                 method=mth)

                if (NndctOption.nndct_stat.value > 2):
                    quant_efficiency, sqnr = quant_data.quant_efficiency(blob.cpu().detach().numpy(), 8)
                    torchquantizer.global_snr_inv += 1 / sqnr
                    print(
                        f"quant_efficiency={quant_efficiency}, global_snr_inv={torchquantizer.global_snr_inv} {quant_data._name}\n")

                # update param to nndct graph
                if tensor_type == 'param':
                    instance.update_param_to_nndct(node, name, res.cpu().detach().numpy())

                if blob_save is not None:
                    blob_save.values.data = blob
                    blob = blob_save
                    res = blob_save

                return res

            _quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
            _quantizer.do_quantize = do_quantize.__get__(_quantizer)
        quant_model = quantizer.quant_model
        ori_forward = quant_model.forward
        post_method = model.m.method

        def forward(*args, **kwargs):
            out = ori_forward(*args, **kwargs)
            return post_method(out)

        quant_model.forward = forward
        model = quant_model
        model.cuda()
    total = len(dataloader)
    if dump_xmodel:
        total = 1
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

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred


            # Evaluate
            correct = torch.zeros(
                predn.shape[0],
                niou,
                dtype=torch.bool,
                device=device)
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                ctmp = torch.zeros(predn.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
                iou = box_ious(labelsn[:, 1:], predn[:, :4])
                pi = torch.where(
                    (iou >= iouv[0]) & (labelsn[:, 0:1] == predn[:, 5]))  # IoU above threshold and classes match
                if pi[0].shape[0]:
                    match = torch.cat((torch.stack(pi, 1), iou[pi[0], pi[1]][:, None]),
                                        1).cpu().detach().numpy()  # [label, detection, iou]
                    if pi[0].shape[0] > 1:
                        match = match[match[:, 2].argsort()[::-1]]
                        match = match[np.unique(match[:, 1], return_index=True)[1]]
                        match = match[np.unique(match[:, 0], return_index=True)[1]]
                    match = torch.Tensor(match).to(iouv.device)
                    ctmp[match[:, 1].long()] = match[:, 2:3] >= iouv
                correct = ctmp

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

        if dump_xmodel:
            break

    if dump_xmodel:
        return


    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(
            1)
        # ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=class_num)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  + \
        (imgsz, imgsz, batch_size)  # tuple
    print(
        'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' %
        t)


    # Save JSON
    if save_coco and len(jdict):
        # w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(cfg.data_path) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = os.path.join(save_dir, 'val_predictions.json')  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    # Return results
    model.float()  # for training

    maps = np.zeros(class_num) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    logging.info(f'''Test result:
        Batch size:  {cfg.batch}
        mAP: {map}
    ''')
    print('val results: %s' % map)
    if all_result:
        return (mp, mr, map50, map), maps, t
    return map


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

    model = model.cuda()
    evaluate(model,cfg=cfg,device=device)

if __name__ == "__main__":
    main()