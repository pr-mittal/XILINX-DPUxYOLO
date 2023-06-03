#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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

# Copyright (c) Megvii, Inc. and its affiliates.

import os
if os.environ["W_QUANT"]=='1':
    from pytorch_nndct.apis import torch_quantizer

import argparse
import cv2
import torch
from tqdm import tqdm
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess, is_parallel

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX-KITTI prediction!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./dataset/KITTI/training/image_2", help="path to images")
    parser.add_argument("--pred_dir", default="./data/format_kitti2d", help="path to images")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, nargs='+', type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
        quant_dir='quantized',
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

        if os.environ["W_QUANT"]=='1':
            if device == 'gpu':
                device = torch.device('cuda')
            dummy_input = torch.randn([1, 3, 384, 1248]).to(device)
            quantizer = torch_quantizer('test', model, dummy_input, output_dir=quant_dir, device=device)
            quant_model = quantizer.quant_model
            quant_model.eval()
            self.quant_model = quant_model

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():

            if os.environ["W_QUANT"]=='1':
                outputs = self.quant_model(img)
            else:
                outputs = self.model(img)

            if is_parallel(self.model):
                outputs = self.model.module.head.postprocess(outputs)
            else:
                outputs = self.model.head.postprocess(outputs)

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def format_result(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return ""
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        fmt_str = ""
        template = "{} -10 -10 -10 {:.2f} {:.2f} {:.2f} {:.2f} -1000 -1000 -1000 -1000 -1000 -1000 -1000 {:.4f}\n"
        for i in range(len(output)):
            fmt_str += template.format(self.cls_names[int(cls[i])], *bboxes[i].tolist(), float(scores[i]))

        return fmt_str


def image_demo(predictor, path, save_folder):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    os.makedirs(save_folder, exist_ok=True)
    for image_name in tqdm(files):
        outputs, img_info = predictor.inference(image_name)
        fmt_str = predictor.format_result(outputs[0], img_info, predictor.confthre)
        save_file_name = os.path.join(save_folder, os.path.basename(image_name).replace('.png', '.txt'))
        with open(save_file_name, 'wt') as fw:
            fw.write(fmt_str)
    print("All predictions are converted to KITTI format in {}".format(save_folder))


def main(exp, args):
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    # file_name = os.path.join(exp.output_dir, args.experiment_name)
    # os.makedirs(file_name, exist_ok=True)

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        if isinstance(args.tsize, int):
            exp.test_size = (args.tsize, args.tsize)
        elif isinstance(args.tsize, (list, tuple)):
            exp.test_size = args.tsize

    model = exp.get_model()

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    print("Loading checkpoint from {}".format(args.ckpt))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    trt_file = None
    decoder = None

    KITTI_CLASSES = ('Car',)
    predictor = Predictor(
        model, exp, KITTI_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    image_demo(predictor, args.path, args.pred_dir)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
