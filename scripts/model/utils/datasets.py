from torch.utils.data import Dataset
import warnings
import random
import torch
import os
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import Albumentations, xywhn2xyxy, xyxy2xywhn, random_affine

# Parameters
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']
NUM_THREADS = 8

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def check_im_lb(p):
    # Verify one image-label pair
    img_path,label_path = p
    num_miss, num_found, msg = 0, 0, ''

    # verify images
    shape = exif_size(Image.open(img_path))  # image size

    # verify labels
    if os.path.isfile(label_path):
        num_found = 1  # label found
        with open(label_path, 'r') as f:
            l = [x.split() for x in f.read().strip().splitlines() if len(x)]
            l = np.array(l, dtype=np.float32)
    else:
        num_miss = 1  # label missing
        l = np.zeros((0, 5), dtype=np.float32)
    return img_path, l, shape, [], num_miss, num_found, 0, 0, msg

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

# class ListDataset(Dataset):
#     def __init__(self, list_path, img_size=640, multiscale=True, transform=None):
#         self.img_files=os.listdir(list_path+"/images")
#         # self.label_files = []
#         # for path in self.img_files:
#         #     image_dir = os.path.dirname(path)
#         #     label_dir = "labels".join(image_dir.rsplit("images", 1))
#         #     assert label_dir != image_dir, \
#         #         f"Image path must contain a folder named 'images'! \n'{image_dir}'"
#         #     label_file = os.path.join(label_dir, os.path.basename(path))
#         #     label_file = os.path.splitext(label_file)[0] + '.txt'
#         #     self.label_files.append(label_file)
#         self.path=list_path
#         self.img_size = img_size
#         self.max_objects = 100
#         self.multiscale = multiscale
#         self.min_size = self.img_size - 3 * 32
#         self.max_size = self.img_size + 3 * 32
#         self.batch_count = 0
#         self.transform = transform

#     def __getitem__(self, index):

#         # ---------
#         #  Image
#         # ---------
#         try:
#             img_path = self.path+"/images/"+self.img_files[index% len(self.img_files)]
#             img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
#         except Exception:
#             print(f"Could not read image '{img_path}'.")
#             return

#         # ---------
#         #  Label
#         # ---------
#         try:
#             label_path = self.path+"/labels/"+self.img_files[index].replace("jpg","txt")
#             # Ignore warning if file is empty
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 boxes = np.loadtxt(label_path,dtype=np.float32).reshape(-1, 5)
#         except Exception:
#             print(f"Could not read label '{label_path}'.")
#             return

#         # -----------
#         #  Transform
#         # -----------
#         if self.transform:
#             try:
#                 img, bb_targets = self.transform((img, boxes))
#             except Exception:
#                 print("Could not apply transform.")
#                 return

#         return img_path, img, bb_targets

#     def collate_fn(self, batch):
#         self.batch_count += 1

#         # Drop invalid images
#         batch = [data for data in batch if data is not None]

#         paths, imgs, bb_targets = list(zip(*batch))

#         # Selects new image size every tenth batch
#         if self.multiscale and self.batch_count % 10 == 0:
#             self.img_size = random.choice(
#                 range(self.min_size, self.max_size + 1, 32))

#         # Resize images to input shape
#         imgs = torch.stack([resize(img, self.img_size) for img in imgs])

#         # Add sample index to targets
#         for i, boxes in enumerate(bb_targets):
#             boxes[:, 0] = i
#         bb_targets = torch.cat(bb_targets, 0)

#         return paths, imgs, bb_targets

#     def __len__(self):
#         return len(self.img_files)
class Yolo_dataset(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 stride=32, pad=0.0):
        try:
            path = str(Path(path))
            try:
                self.img_files=[path+"/images/"+x for x in os.listdir(path+"/images") if '.'+x.split('.')[-1] in img_formats]
            except Exception:
                raise Exception('Error loading data')
            # parent = str(Path(path).parent) + os.sep
            # if os.path.isfile(path):  # file
            #     with open(path, 'r') as f:
            #         f = f.read().strip().splitlines()
            #         f = [x.replace('./', parent) if x.startswith('./') else x for x in f]
            # self.img_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except:
            raise Exception('Error loading data')

        n = len(self.img_files)
        assert n > 0, 'No images found'
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.indices = range(n)
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.albumentations = Albumentations() if augment else None
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # Check cache
        cache_path = (Path(path) if Path(path).is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        # cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
        except:
            cache, exists = self.save_lb_cache(cache_path), False  # cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=d, total=n, initial=n)  # display cache results
        # print(len(cache.values()))
        # for x in cache.values():
        #     print(len(x))
        # print(cache.values())
        # Read cache
        labels=[]
        shapes=[] 
        self.segments=[]
        self.img_files=[]
        for y,x in cache.items():
            if(type(x)==list and len(x)==3):
                labels.append(x[0])
                shapes.append(x[1])
                self.segments.append(x[2])
                self.img_files.append(y)
        # zip(*cache.values())
        # print(cache.keys())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        # self.img_files = list(cache.keys())  # update

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        # self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            # for x in cache.keys()]  # update

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        self.img_npy = [None] * n
    def save_lb_cache(self, path=Path('./labels.cache')):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        num_miss, num_found, num_empty, num_corrupt, msgs = 0, 0, 0, 0, []
        desc = f"Scanning '{path.parent / path.stem}' images and labels..."
        # print(self.img_files, self.label_files)
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(check_im_lb, zip(self.img_files, self.label_files)),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                num_miss += nm_f
                num_found += nf_f
                num_empty += ne_f
                num_corrupt += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{num_miss} found, {num_found} missing, {num_empty} empty, {num_corrupt} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        x['results'] = num_found, num_miss, num_empty, num_corrupt, len(self.img_files)
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            logging.info(f'cache file created: {path}')
        except Exception as e:
            logging.info(f'WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x
    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # # MixUp augmentation
            # if random.random() < hyp['mixup']:
            #     img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_affine(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v)
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached
        npy = self.img_npy[i]
        if npy and npy.exists():
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, 'Image Not Found ' + path
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


def load_mosaic(self, index):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine


    # Augment
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=self.mosaic_border)  # border to remove
    return img4, labels4
