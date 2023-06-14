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

import time
import logging
import os, sys, math
import argparse
import datetime


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from easydict import EasyDict as edict
from copy import deepcopy
from utils.datasets import Yolo_dataset
from cfg import Cfg
from torch.cuda import amp
from models.models import ofa_yolo_0
from functools import partial

import yaml
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from utils.utils import labels_to_class_weights,check_anchors,compute_loss,fitness
from utils import torch_utils
from yolov3_test import evaluate
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def train(model, device, cfg,nndct_quant=False):
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    epochs = cfg.epoch
    batch_size = cfg.batchsize
    lr0 = cfg.lr0
    lrf = cfg.lrf
    qat_group = cfg.qat_group
    cuda = device.type != 'cpu'
    imgsz = cfg.imgsz
    conf_thres = cfg.conf_thres
    iou_thres = cfg.iou_thres
    nw = cfg.num_worker  # number of workers
    if not os.path.exists(cfg.save_dir):
        os.mkdir(cfg.save_dir)
    weight_dir = cfg.save_dir + '/weight'  # increment run
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    last_model = weight_dir+'/last_model.pt'
    last_model_ema = weight_dir+'/last_model_ema.pt'
    best_model = weight_dir+'/best_model.pt'
    best_model_ema = weight_dir+'/best_model_ema.pt'

    # Save run settings
    import json
    with open(cfg.save_dir + '/cfg.json', 'w') as f:
        json.dump(cfg,f,indent=4)

    # Dataloader
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once pad = 0.0 if task == 'speed' else 0.5
    class_num = cfg.class_num
    train_image = cfg.data_path + cfg.train_image
    dataset = Yolo_dataset(train_image, imgsz, batch_size, hyp=cfg, augment=True, rect=cfg.rect)
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    nb = len(dataloader)
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr0}
        Training size:   {nb}
        Checkpoints:     {weight_dir}
        Device:          {device.type}
        Images size:     {imgsz}
        Optimizer:       {cfg.TRAIN_OPTIMIZER}
        Dataset classes: {class_num}
        nndct_quant:     {nndct_quant}
    ''')
    # evaluate(model=model,
    #          cfg=cfg,
    #          device=device,
    #          half=not nndct_quant,
    #          nndct_quant=False,
    #          all_result=True)
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    cfg.weight_decay *= batch_size * accumulate / nbs  # scale weight_decay
    logging.info(f"Scaled weight_decay = {cfg.weight_decay}")
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if cfg.TRAIN_OPTIMIZER == 'adam':
        optimizer = Adam(g0, lr=lr0, betas=(cfg.momentum, 0.999))  # adjust beta1 to momentum
    elif cfg.TRAIN_OPTIMIZER == 'adamw':
        optimizer = AdamW(g0, lr=lr0, betas=(cfg.momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=lr0, momentum=cfg.momentum, nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': cfg.weight_decay})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    
    threshold = [
        param for name, param in model.named_parameters()
        if 'threshold' in name
    ]
    g_qat = {
        'params': threshold,
        'lr': lr0 * 100,
        'name': 'threshold'
    }
    optimizer.add_param_group(g_qat)
    logging.info(f"optimizer: {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias, {len(g_qat)} log_threshold")
    del g0, g1, g2, g_qat
   
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    start_epoch, best_fitness = 0, 0.0
    best_epoch = 0
    lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf-1)+ 1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # do not move

    # Model parameters
    cfg.cls *= class_num / 7.   # scale to classes and layers
    cfg.obj *= (imgsz / 640) ** 2  # scale to image size and layers
    model.nc = class_num  # attach number of classes to model
    model.hyp = cfg  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, class_num).to(device) * class_num  # attach class weights
    model.names = cfg.class_names

    # Check anchors
    # import pdb;pdb.set_trace()
    check_anchors(dataset, model=model, thr=cfg.anchor_t, imgsz=imgsz)

    ema = torch_utils.ModelEMA(model)


    # Start training
    t0 = time.time()
    nb = len(dataloader)  # number of batches
    maps = np.zeros(class_num)  # mAP per class
    n_burn = max(round(cfg.warmup_epochs* nb), 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scaler = amp.GradScaler(enabled=cuda and not nndct_quant)
    last_opt_step = -1
    print('Image sizes %g train' % (imgsz))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    for epoch in range(start_epoch,epochs): #2): # epoch ------------------------------------------------------------------
        model.train()
        #freeze model if epochs>200
        if(epochs>=60 and epoch==start_epoch):
            model.freeze()
        elif(epoch==30):
            model.unfreeze()
        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)
        logging.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [cfg.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [cfg.warmup_momentum, cfg.momentum])
            with amp.autocast(enabled=cuda and not nndct_quant):
                pred = model(imgs)  # forward
                # print("PRED",pred,"TARGET",targets)
                loss, loss_items = compute_loss(pred, targets.to(device).float(), model, None)
            # Backward
            scaler.scale(loss).backward()
            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Print
            # print("PRINT",mloss,i,loss_items)
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            # mloss_list=[]
            # for x in mloss:
            #     mloss_list.append(x.item())
            # print(mloss_list)
            s = ('%10s' * 2 + '%10.4g' * 5) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            exit
            # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        scheduler.step()
        logging.info(s)

        # mAP
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs

        if not cfg.notest or final_epoch:  # Calculate mAP
            results, maps, times = evaluate(model=ema.ema,
                                            cfg=cfg,
                                            device=device,
                                            half=True,
                                            nndct_quant=False,
                                            all_result=True,
                                            train_val=True,
                                            deploy_dir='',
                                            save_coco=False
                                            )

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi
            best_epoch = epoch


        # Save model
        save = (not cfg.notest) or final_epoch
        if save:
            epoch_path = weight_dir + '/' + str(epoch) + '.pt'
            epoch_path_ema = weight_dir + '/' + str(epoch) + '_ema.pt'
            ckpt = model.model.state_dict()
            ckpt['anchors']=model.m.anchors
            ckpt_ema = ema.ema.model.state_dict()
            ckpt_ema['anchors']=ema.ema.m.anchors
            logging.info(f'Results:\n\tepoch: {epoch}\n\tmAP:  {results[-1]}\n\tfitness: {fi}')
            # Save last, best and delete
            torch.save(ckpt, last_model)
            torch.save(ckpt_ema, last_model_ema)

            if best_fitness == fi:
                logging.info(f'Best results update:\n\tepoch: {epoch}\n\tmAP:  {results[-1]}\n\tfitness: {fi}')
                torch.save(ckpt, best_model)
                torch.save(ckpt_ema, best_model_ema)

            if (epoch >= 0) and (cfg.save_period > 0) and (epoch % cfg.save_period == 0):
                torch.save(ckpt, epoch_path)
                torch.save(ckpt, epoch_path_ema)


            del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    logging.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    torch.cuda.empty_cache()
    return

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-e', '--epoch', type=int, default=420)
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0',help='GPU', dest='gpu')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument('--save-dir', default='./build/train', help='save to save_dir')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--nndct_quant', action='store_true', help='Train nndct QAT model')
    parser.add_argument('--qat_group', action='store_true', help='param groups')
    parser.add_argument('--ratio', default=0, help='pruning ratio')
    parser.add_argument('--num_worker', default=8, help='number of workers')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'], help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
    parser.add_argument('--dump_xmodel', action='store_true', default=False)
    parser.add_argument('--fast_finetune', action='store_true', default=False)
    parser.add_argument('--qt_dir', default='./build/nndct_quant', help='save to quant_info')
    parser.add_argument('--nndct_bitwidth', type=int, default=8, help='save to quant_info')
    parser.add_argument('--log_dir', default='./build/log', help='save to log')

    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)


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


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    cfg = get_args(**Cfg)
    logging = init_logger(log_dir=cfg.log_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if(torch.cuda.is_available()):
        torch.cuda.empty_cache()
        force_cudnn_initialization()

    # # model
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    if cfg.pretrained:
        pretrained_ofa_model = cfg.pretrained 
        with open(pretrained_ofa_model, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        anchors_weight = checkpoint['anchors']
        if cfg.ratio == 0:
            model = ofa_yolo_0(anchors, anchors_weight)
        # model = Model(anchors, anchors_weight)
        del checkpoint['anchors']
        model.model.load_state_dict(checkpoint, strict=True)
        print(f'load successfully: ratio:{cfg.ratio};  pretrained:{pretrained_ofa_model}')
        model = model.to(device)
    else:
        if cfg.ratio == 0:
            model = ofa_yolo_0(anchors)
        model = model.to(device)
    # model.freeze()
    # exit(0)
    try:
        if cfg.nndct_quant:
            train(model=model,
                  device=device,
                  cfg=cfg,
                  nndct_quant=True)
        else:
            train(model=model,
              device=device,
              cfg=cfg)
    except KeyboardInterrupt:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), 'INTERRUPTED.pth')
        else:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
