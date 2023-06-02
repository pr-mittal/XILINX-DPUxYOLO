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
import datetime
from copy import deepcopy
import torch
from functools import partial
from cfg import Cfg
from easydict import EasyDict as edict
from test import evaluate
from pytorch_nndct.apis import torch_quantizer
from pytorch_nndct import QatProcessor
from nndct_shared.utils import NndctOption, option_util, NndctDebugLogger, NndctScreenLogger
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
    parser.add_argument('--dump_xmodel', action='store_true', default=False)
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)

def main():
    logging = init_logger(log_dir='log')
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
    ema_weight = checkpoint['qat_ema_state_dict']

    if cfg.ratio == 30:
        model = ofa_yolo_30(anchors)
    elif cfg.ratio == 0:
        model = ofa_yolo_0(anchors)
    elif cfg.ratio == 50:
        model = ofa_yolo_50(anchors)
    model.cuda()


    _ori_model = deepcopy(model)
    model.train()
    im = (torch.zeros(1, 3, cfg.imgsz, cfg.imgsz).to(device).type_as(next(model.parameters())))
    for _ in range(2):
        y = model(im)  # dry runs
    model.forward = partial(model.forward, quant=True)
    qat_processor = QatProcessor(model, (im,), bitwidth=8, mix_bit=False)
    _trainable_model = qat_processor.trainable_model()
    _trainable_model.load_state_dict(ema_weight, strict=True)
    deploy_path = f'./tmp_deployable_model_{cfg.ratio}'
    _deployable_net = qat_processor.convert_to_deployable(_trainable_model, deploy_path)
    _ori_model.load_state_dict(_deployable_net.state_dict(), strict=True)
    model=_ori_model
    print('load successfully')
    model.eval()
    model.cuda()
    from pytorch_nndct.apis import torch_quantizer
    import pytorch_nndct as py_nndct
    from nndct_shared.utils import NndctOption
    from nndct_shared.base import key_names, NNDCT_KEYS, NNDCT_DEBUG_LVL, GLOBAL_MAP, NNDCT_OP
    import nndct_shared.quantization as nndct_quant
    from pytorch_nndct.quantization import torchquantizer
    model.forward = partial(model.forward, quant=True)
    print(f"NNDCT quant dir: {deploy_path}")
    quantizer = torch_quantizer(quant_mode='test',
                                bitwidth=8,
                                module=model,
                                input_args=im,
                                output_dir=os.path.join(deploy_path, 'test'))
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

    def forward(x):
        out = ori_forward(x)
        return post_method(out)

    quant_model.forward = forward

    model = quant_model
    if cfg.dump_xmodel:
        cfg.batchsize = 1
        evaluate(model, cfg=cfg, device=device, nndct_quant=True, dump_xmodel=True)
        quantizer.export_xmodel(deploy_path, deploy_check=True)
    else:
        evaluate(model, cfg=cfg, device=device, nndct_quant=True)


if __name__ == "__main__":
    main()