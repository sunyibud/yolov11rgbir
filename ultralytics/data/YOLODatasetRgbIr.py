import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str
from utils.torch_utils import torch_distributed_zero_first

import global_var

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)


# FQY 构建随机采样的sampler，为了之后双模态输入
class RandomSampler(torch.utils.data.sampler.RandomSampler):

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # print("-------------------------")
        s = global_var.get_value('s')
        return iter(s)

    def __len__(self):
        return self.num_samples


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


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


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def random_perspective_rgb_ir(img_rgb, img_ir, targets_rgb=(), targets_ir=(), segments_rgb=(), segments_ir=(),
                              degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    img = img_rgb
    targets = targets_rgb
    segments = segments_rgb

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            # img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            img_rgb = cv2.warpPerspective(img_rgb, M, dsize=(width, height), borderValue=(114, 114, 114))
            img_ir = cv2.warpPerspective(img_ir, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            # img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img_rgb = cv2.warpAffine(img_rgb, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img_ir = cv2.warpAffine(img_ir, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img_rgb[:, :, ::-1])  # base
    # ax[1].imshow(img_ir[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img_rgb, img_ir, targets, targets


def load_image_rgb_ir(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw

    img_rgb = self.imgs_rgb[index]
    img_ir = self.imgs_ir[index]
    # img_rgb = None
    # img_ir = None

    if (img_rgb is None) and (img_ir is None):  # not cached

        path_rgb = self.img_files_rgb[index]
        path_ir = self.img_files_ir[index]

        # # print("load_image_rgb_ir")
        # print(path_rgb)
        # print(path_ir)

        img_rgb = cv2.imread(path_rgb)  # BGR
        img_ir = cv2.imread(path_ir)  # BGR

        assert img_rgb is not None, 'Image RGB Not Found ' + path_rgb
        assert img_ir is not None, 'Image IR Not Found ' + path_ir

        h0, w0 = img_rgb.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img_rgb = cv2.resize(img_rgb, (int(w0 * r), int(h0 * r)),
                                 interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            img_ir = cv2.resize(img_ir, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img_rgb, img_ir, (h0, w0), img_rgb.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs_rgb[index], self.imgs_ir[index], self.img_hw0_rgb[index], self.img_hw_rgb[
            index]  # img, hw_original, hw_resized


def load_mosaic_RGB_IR(self, index1, index2):
    index_rgb = index1
    index_ir = index2

    labels4_rgb, segments4_rgb = [], []
    labels4_ir, segments4_ir = [], []

    s = self.img_size

    # print("image size ", s)

    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y

    assert index_rgb == index_ir, 'INDEX RGB 不等于 INDEX IR'

    indices = [index_rgb] + random.choices(self.indices_rgb, k=3)  # 3 additional image indices

    for i, index in enumerate(indices):
        # Load image

        # img, _, (h, w) = load_image(self, index)
        img_rgb, img_ir, _, (h, w) = load_image_rgb_ir(self, index)

        # place img in img4
        if i == 0:  # top left
            img4_rgb = np.full((s * 2, s * 2, img_rgb.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            img4_ir = np.full((s * 2, s * 2, img_ir.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
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

        img4_rgb[y1a:y2a, x1a:x2a] = img_rgb[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        img4_ir[y1a:y2a, x1a:x2a] = img_ir[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        labels_rgb, segments_rgb = self.labels_rgb[index].copy(), self.segments_rgb[index].copy()
        labels_ir, segments_ir = self.labels_ir[index].copy(), self.segments_ir[index].copy()
        if labels_rgb.size:
            labels_rgb[:, 1:] = xywhn2xyxy(labels_rgb[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels_ir[:, 1:] = xywhn2xyxy(labels_ir[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments_rgb = [xyn2xy(x, w, h, padw, padh) for x in segments_rgb]
            segments_ir = [xyn2xy(x, w, h, padw, padh) for x in segments_ir]
        labels4_rgb.append(labels_rgb)
        segments4_rgb.extend(segments_rgb)
        labels4_ir.append(labels_ir)
        segments4_ir.extend(segments_ir)


    labels4_rgb = np.concatenate(labels4_rgb, 0)
    labels4_ir = np.concatenate(labels4_ir, 0)
    for x in (labels4_rgb[:, 1:], *segments4_rgb):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    for x in (labels4_ir[:, 1:], *segments4_ir):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()


    img4_rgb, img4_ir, labels4_rgb, labels4_ir = random_perspective_rgb_ir(img4_rgb, img4_ir, labels4_rgb, labels4_ir,
                                                                           segments4_rgb, segments4_ir,
                                                                           degrees=self.hyp['degrees'],
                                                                           translate=self.hyp['translate'],
                                                                           scale=self.hyp['scale'],
                                                                           shear=self.hyp['shear'],
                                                                           perspective=self.hyp['perspective'],
                                                                           border=self.mosaic_border)  # border to remove

    # print(labels_rgb)
    # print(labels4_ir)

    # assert labels4_rgb == labels4_ir, 'LABEL4 RGB 不等于 LABEL4 IR'
    # print(" labels4_rgb == labels4_ir ", labels4_rgb == labels4_ir)
    labels4_ir = labels4_rgb

    # cv2.imwrite("rgb_%s.jpg" % str(index1), img4_rgb)
    # cv2.imwrite("ir_%s.jpg" % str(index2), img4_ir)

    return img4_rgb, labels4_rgb, img4_ir, labels4_ir


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
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
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class YOLODatasetRgbIr(Dataset):
    def __init__(self, img_rgb_path, img_ir_path, imgsz=640, batch_size=16,
                 augment=False, hyp=None, rect=False, cache=False,
                 single_cls=False, stride=32, pad=0.0, prefix='', task=None,
                 classes=None, data=None, fraction=None, image_weights=False):
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-imgsz // 2, -imgsz // 2]
        self.stride = stride
        self.img_rgb_path = img_rgb_path
        self.img_ir_path = img_ir_path

        try:
            f_rgb = []  # image files
            f_ir = []
            # -----------------------------  rgb   -----------------------------
            for p_rgb in img_rgb_path if isinstance(img_rgb_path, list) else [img_rgb_path]:
                p_rgb = Path(p_rgb)  # os-agnostic
                if p_rgb.is_dir():  # dir
                    f_rgb += glob.glob(str(p_rgb / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_rgb.is_file():  # file
                    with open(p_rgb, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_rgb.parent) + os.sep
                        f_rgb += [x.replace('./', parent) if x.startswith('./') else x for x in
                                  t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{img_rgb_path} does not exist')

                # -----------------------------  ir   -----------------------------
                for p_ir in img_ir_path if isinstance(img_ir_path, list) else [img_ir_path]:
                    p_ir = Path(p_ir)  # os-agnostic
                    if p_ir.is_dir():  # dir
                        f_ir += glob.glob(str(p_ir / '**' / '*.*'), recursive=True)
                        # f = list(p.rglob('**/*.*'))  # pathlib
                    elif p_ir.is_file():  # file
                        with open(p_ir, 'r') as t:
                            t = t.read().strip().splitlines()
                            parent = str(p_ir.parent) + os.sep
                            f_ir += [x.replace('./', parent) if x.startswith('./') else x for x in
                                     t]  # local to global path
                            # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                    else:
                        raise Exception(f'{prefix}{p_ir} does not exist')

            self.img_files_rgb = sorted(
                [x.replace('/', os.sep) for x in f_rgb if x.split('.')[-1].lower() in img_formats])
            self.img_files_ir = sorted(
                [x.replace('/', os.sep) for x in f_ir if x.split('.')[-1].lower() in img_formats])

            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert (self.img_files_rgb, self.img_files_ir), (f'{prefix}No images found', f'{prefix}No images found')
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {img_rgb_path, img_ir_path}: {e}\nSee {help_url}')

        # Check cache
        # Check rgb cache
        self.label_files_rgb = img2label_paths(self.img_files_rgb)  # labels
        # print(self.label_files)
        cache_rgb_path = (p_rgb if p_rgb.is_file() else Path(self.label_files_rgb[0]).parent).with_suffix(
            '.cache')  # cached labels
        if cache_rgb_path.is_file():
            cache_rgb, exists_rgb = torch.load(cache_rgb_path, weights_only=False), True  # load
            if cache_rgb['hash'] != get_hash(
                    self.label_files_rgb + self.img_files_rgb) or 'version' not in cache_rgb:  # changed
                cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb, self.label_files_rgb,
                                                          cache_rgb_path, prefix), False  # re-cache
        else:
            cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb, self.label_files_rgb,
                                                      cache_rgb_path, prefix), False  # cache

        # Check ir cache
        self.label_files_ir = img2label_paths(self.img_files_ir)  # labels
        # print(self.label_files)
        cache_ir_path = (p_ir if p_ir.is_file() else Path(self.label_files_ir[0]).parent).with_suffix(
            '.cache')  # cached labels
        if cache_ir_path.is_file():
            cache_ir, exists_ir = torch.load(cache_ir_path, weights_only=False), True  # load
            if cache_ir['hash'] != get_hash(
                    self.label_files_ir + self.img_files_ir) or 'version' not in cache_ir:  # changed
                cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir,
                                                        cache_ir_path, prefix), False  # re-cache
        else:
            cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir,
                                                    cache_ir_path, prefix), False  # cache

        # Display cache
        nf_rgb, nm_rgb, ne_rgb, nc_rgb, n_rgb = cache_rgb.pop('results')  # found, missing, empty, corrupted, total
        nf_ir, nm_ir, ne_ir, nc_ir, n_ir = cache_ir.pop('results')  # found, missing, empty, corrupted, total
        if exists_rgb:
            d = f"Scanning RGB '{cache_rgb_path}' images and labels... {nf_rgb} found, {nm_rgb} missing, {ne_rgb} empty, {nc_rgb} corrupted"
            tqdm(None, desc=prefix + d, total=n_rgb, initial=n_rgb)  # display cache results
        if exists_ir:
            d = f"Scanning IR '{cache_rgb_path}' images and labels... {nf_ir} found, {nm_ir} missing, {ne_ir} empty, {nc_ir} corrupted"
            tqdm(None, desc=prefix + d, total=n_ir, initial=n_ir)  # display cache results

        assert nf_rgb > 0 or not augment, f'{prefix}No labels in {cache_rgb_path}. Can not train without labels. See {help_url}'

        # Read cache
        # Read RGB cache
        cache_rgb.pop('hash')  # remove hash
        cache_rgb.pop('version')  # remove version
        labels_rgb, shapes_rgb, self.segments_rgb = zip(*cache_rgb.values())
        self.labels_rgb = list(labels_rgb)
        self.shapes_rgb = np.array(shapes_rgb, dtype=np.float64)
        self.img_files_rgb = list(cache_rgb.keys())  # update
        self.label_files_rgb = img2label_paths(cache_rgb.keys())  # update
        if single_cls:
            for x in self.labels_rgb:
                x[:, 0] = 0

        n_rgb = len(shapes_rgb)  # number of images
        bi_rgb = np.floor(np.arange(n_rgb) / batch_size).astype(np.int64)
        nb_rgb = bi_rgb[-1] + 1  # number of batches
        self.batch_rgb = bi_rgb  # batch index of image
        self.n_rgb = n_rgb
        self.indices_rgb = range(n_rgb)

        # Read IR cache
        cache_ir.pop('hash')  # remove hash
        cache_ir.pop('version')  # remove version
        labels_ir, shapes_ir, self.segments_ir = zip(*cache_ir.values())
        self.labels_ir = list(labels_ir)
        self.shapes_ir = np.array(shapes_ir, dtype=np.float64)
        self.img_files_ir = list(cache_ir.keys())  # update
        self.label_files_ir = img2label_paths(cache_ir.keys())  # update
        if single_cls:
            for x in self.labels_ir:
                x[:, 0] = 0

        n_ir = len(shapes_ir)  # number of images
        bi_ir = np.floor(np.arange(n_ir) / batch_size).astype(np.int64)  # batch index
        nb_ir = bi_ir[-1] + 1  # number of batches
        self.batch_ir = bi_ir  # batch index of image
        self.n_ir = n_ir
        self.indices_ir = range(n_ir)

        # Rectangular Training
        if self.rect:

            # RGB
            # Sort by aspect ratio
            s_rgb = self.shapes_rgb  # wh
            ar_rgb = s_rgb[:, 1] / s_rgb[:, 0]  # aspect ratio
            irect_rgb = ar_rgb.argsort()
            self.img_files_rgb = [self.img_files_rgb[i] for i in irect_rgb]
            self.label_files_rgb = [self.label_files_rgb[i] for i in irect_rgb]
            self.labels_rgb = [self.labels_rgb[i] for i in irect_rgb]
            self.shapes_rgb = s_rgb[irect_rgb]  # wh
            ar_rgb = ar_rgb[irect_rgb]

            # Set training image shapes
            shapes_rgb = [[1, 1]] * nb_rgb
            for i in range(nb_rgb):
                ari_rgb = ar_rgb[bi_rgb == i]
                mini, maxi = ari_rgb.min(), ari_rgb.max()
                if maxi < 1:
                    shapes_rgb[i] = [maxi, 1]
                elif mini > 1:
                    shapes_rgb[i] = [1, 1 / mini]

            self.batch_shapes_rgb = np.ceil(np.array(shapes_rgb) * imgsz / stride + pad).astype(np.int64) * stride

            # IR
            # Sort by aspect ratio
            s_ir = self.shapes_ir  # wh
            ar_ir = s_ir[:, 1] / s_ir[:, 0]  # aspect ratio
            irect_ir = ar_ir.argsort()
            self.img_files_ir = [self.img_files_ir[i] for i in irect_ir]
            self.label_files_ir = [self.label_files_ir[i] for i in irect_ir]
            self.labels_ir = [self.labels_ir[i] for i in irect_ir]
            self.shapes_ir = s_ir[irect_ir]  # wh
            ar_ir = ar_ir[irect_ir]

            # Set training image shapes
            shapes_ir = [[1, 1]] * nb_ir
            for i in range(nb_ir):
                ari_ir = ar_ir[bi_ir == i]
                mini, maxi = ari_ir.min(), ari_ir.max()
                if maxi < 1:
                    shapes_ir[i] = [maxi, 1]
                elif mini > 1:
                    shapes_ir[i] = [1, 1 / mini]

            self.batch_shapes_ir = np.ceil(np.array(shapes_ir) * imgsz / stride + pad).astype(np.int64) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs_rgb = [None] * n_rgb
        self.imgs_ir = [None] * n_ir
        self.labels = self.labels_rgb
        self.shapes = self.shapes_rgb
        self.indices = self.indices_rgb

    def cache_labels(self, imgfiles, labelfiles, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        img_files = imgfiles
        label_files = labelfiles
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(img_files, label_files), desc='Scanning images', total=len(img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(label_files + img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files_rgb)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        # index = self.indices[index]  # linear, shuffled, or image_weights
        index_rgb = self.indices_rgb[index]  # linear, shuffled, or image_weights
        index_ir = self.indices_ir[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic

            # img, labels = load_mosaic(self, index)
            img_rgb, labels_rgb, img_ir, labels_ir = load_mosaic_RGB_IR(self, index_rgb, index_ir)

            # # FQY 打印图片
            # # print("--------------------------------------Load mosaic")
            # im_rgb = Image.fromarray(img_rgb.astype('uint8')).convert('RGB')
            # Image.Image.save(im_rgb, 'example_%s_%s.jpg' % (str(index), "RGB"))
            # print(' write the example_%s_%s.jpg' % (self.img_rgb_path.split("/")[7], str(index)))
            # im_ir = Image.fromarray(img_ir.astype('uint8')).convert('RGB')
            # Image.Image.save(im_ir, 'example_%s_%s.jpg' % (str(index), "IR"))
            # print(' write the example_%s_%s.jpg' % (self.img_ir_path.split("/")[7], str(index)))

            shapes = None

            # # MixUp https://arxiv.org/pdf/1710.09412.pdf
            # if random.random() < hyp['mixup']:
            #
            #     img2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
            #     r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
            #     img = (img * r + img2 * (1 - r)).astype(np.uint8)
            #     labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img_rgb, img_ir, (h0, w0), (h, w) = load_image_rgb_ir(self, index)

            # Letterbox
            shape = self.batch_shapes_rgb[self.batch_rgb[index]] if self.rect else self.imgsz  # final letterboxed shape
            img_rgb, ratio, pad = letterbox(img_rgb, shape, auto=False, scaleup=self.augment)
            img_ir, ratio, pad = letterbox(img_ir, shape, auto=False, scaleup=self.augment)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels_rgb[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            labels_rgb = labels
            labels_ir = labels

        if self.augment:
            # Augment colorspace
            augment_hsv(img_rgb, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            augment_hsv(img_ir, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
        nL = len(labels_rgb)  # number of labels
        if nL:
            labels_rgb[:, 1:5] = xyxy2xywh(labels_rgb[:, 1:5])  # convert xyxy to xywh
            labels_rgb[:, [2, 4]] /= img_rgb.shape[0]  # normalized height 0-1
            labels_rgb[:, [1, 3]] /= img_rgb.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img_rgb = np.flipud(img_rgb)
                img_ir = np.flipud(img_ir)
                if nL:
                    labels_rgb[:, 2] = 1 - labels_rgb[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img_rgb = np.fliplr(img_rgb)
                img_ir = np.fliplr(img_ir)

                if nL:
                    labels_rgb[:, 1] = 1 - labels_rgb[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels_rgb)

        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = np.ascontiguousarray(img)

        img_rgb = img_rgb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_rgb = np.ascontiguousarray(img_rgb)
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_ir = np.ascontiguousarray(img_ir)

        img_all = np.concatenate((img_rgb, img_ir), axis=0)

        return torch.from_numpy(img_all), labels_out, self.img_files_rgb[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4
