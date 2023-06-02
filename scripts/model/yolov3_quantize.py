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
def test(model,test_loader,class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    '''
    test the model
    '''
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    num_cycles=10
    for i in tqdm.tqdm(range(num_cycles), desc="Validating"):
        _, imgs, targets=next(iter(test_loader))
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

        # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output
def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

def quantize_model(model_path, weights_path, img_path, class_names,
                     batch_size=8, img_size=416, n_cpu=8, iou_thres=0.5 ,conf_thres=0.5, nms_thres=0.5,build_dir="build", quant_mode="calib", verbose=True):
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    dataloader = _create_validation_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    metrics_output = test(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    
    # float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'

    # override batchsize if in test mode
    if (quant_mode=='test'):
        batch_size = 1
    rand_in = torch.randn([batch_size, 3, img_size,img_size])
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
    quantized_model = quantizer.quant_model

    metrics_output = test(
        quantized_model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
      # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
    print("DONE")
    return
def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="yolov3/config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="yolov3/weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-i", "--images", type=str, default="yolov3/data/val2014", help="Path to directory with images to inference")
    parser.add_argument("-c", "--classes", type=str, default="yolov3/data/coco.names", help="Path to classes label file (.names)")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    parser.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
    parser.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names

    quantize_model(
        args.model,
        args.weights,
        args.images,
        classes,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        build_dir=args.build_dir,
        quant_mode=args.quant_mode,
        verbose=True)


if __name__ == '__main__':
    run()