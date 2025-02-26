# -*- coding:utf-8 -*-
import cv2
from ultralytics import YOLO
import os
import argparse
import time
import torch

parser = argparse.ArgumentParser()
# 检测参数
parser.add_argument('--weights', default=r"yolo11n.pt", type=str, help='weights path')
parser.add_argument('--source', default=r"images", type=str, help='img or video(.mp4)path')
parser.add_argument('--save', default=r"./save", type=str, help='save img or video path')
parser.add_argument('--vis', default=True, action='store_true', help='visualize image')
parser.add_argument('--conf_thre', type=float, default=0.2, help='conf_thre')
parser.add_argument('--iou_thre', type=float, default=0.6, help='iou_thre')
opt = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


class Detector(object):
    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5):
        self.device = device
        self.model = YOLO(weight_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.names = self.model.names

    def detect_image(self, img_bgr):
        results = self.model(img_bgr, verbose=True, conf=self.conf_threshold,
                             iou=self.iou_threshold, device=self.device)

        bboxes_cls = results[0].boxes.cls
        bboxes_conf = results[0].boxes.conf
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')

        for idx in range(len(bboxes_cls)):
            box_cls = int(bboxes_cls[idx])
            bbox_xyxy = bboxes_xyxy[idx]
            bbox_label = self.names[box_cls]
            box_conf = f"{bboxes_conf[idx]:.2f}"
            xmax, ymax, xmin, ymin = bbox_xyxy[2], bbox_xyxy[3], bbox_xyxy[0], bbox_xyxy[1]

            img_bgr = cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), get_color(box_cls + 3), 2)
            cv2.putText(img_bgr, f'{str(bbox_label)}/{str(box_conf)}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(box_cls + 3), 2)
        return img_bgr


# Example usage
if __name__ == '__main__':
    model = Detector(weight_path=opt.weights, conf_threshold=opt.conf_thre, iou_threshold=opt.iou_thre)
    images_format = ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG']
    video_format = ['mov', 'MOV', 'mp4', 'MP4']

    if os.path.join(opt.source).split(".")[-1] not in video_format:
        image_names = [name for name in os.listdir(opt.source) for item in images_format if
                       os.path.splitext(name)[1] == item]
        for img_name in image_names:
            img_path = os.path.join(opt.source, img_name)
            img_ori = cv2.imread(img_path)
            img_vis = model.detect_image(img_ori)
            img_vis = cv2.resize(img_vis, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(opt.save, img_name), img_vis)

            if opt.vis:
                cv2.imshow(img_name, img_vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    else:
        capture = cv2.VideoCapture(opt.source)
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        outVideo = cv2.VideoWriter(os.path.join(opt.save, os.path.basename(opt.source).split('.')[-2] + "_out.mp4"),
                                   fourcc,
                                   fps, size)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            start_frame_time = time.perf_counter()
            img_vis = model.detect_image(frame)
            # 结束计时
            end_frame_time = time.perf_counter()  # 使用perf_counter进行时间记录
            # 计算每帧处理的FPS
            elapsed_time = end_frame_time - start_frame_time
            if elapsed_time == 0:
                fps_estimation = 0.0
            else:
                fps_estimation = 1 / elapsed_time

            h, w, c = img_vis.shape
            cv2.putText(img_vis, f"FPS: {fps_estimation:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)

            outVideo.write(img_vis)
            cv2.imshow('detect', img_vis)
            cv2.waitKey(1)

        capture.release()
        outVideo.release()
