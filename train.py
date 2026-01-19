import argparse
import platform
import time
from datetime import datetime
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version


# 可指定的所有参数（对应 ultralytics/cfg/default.yaml 的键；含默认值/注释请看该文件）
# task, mode, model, data, epochs, time, patience, batch, imgsz, save, save_period, cache, device, workers, project,
# name, exist_ok, pretrained, optimizer, verbose, seed, deterministic, single_cls, rect, cos_lr, close_mosaic, resume,
# amp, fraction, profile, freeze, multi_scale, compile, val, split, save_json, conf, iou, max_det, half, dnn, plots,
# source, vid_stride, stream_buffer, visualize, augment, agnostic_nms, classes, embed, show, save_frames, save_txt,
# save_conf, save_crop, show_labels, show_conf, show_boxes, line_width, format, keras, optimize, int8, dynamic,
# simplify, opset, workspace, nms, lr0, lrf, momentum, weight_decay, warmup_epochs, warmup_momentum, warmup_bias_lr,
# box, cls, dfl, nbs, hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective, flipud, fliplr, bgr, mosaic,
# mixup, cutmix, copy_paste, copy_paste_mode, overlap_mask, mask_ratio, auto_augment, erasing, cfg, tracker


def _format_metrics_table(metrics, seen, decimals: int = 6):
    if metrics is None or not hasattr(metrics, "mean_results"):
        return "No validation metrics available."
    names = metrics.names or {}
    total_instances = int(metrics.nt_per_class.sum()) if metrics.nt_per_class is not None else 0
    total_images = int(seen) if seen is not None else 0
    mp, mr, map50, map5095 = metrics.mean_results()
    rows = [("all", total_images, total_instances, mp, mr, map50, map5095)]
    if metrics.ap_class_index is not None and metrics.nt_per_class is not None and metrics.nt_per_image is not None:
        for i, c in enumerate(metrics.ap_class_index):
            rows.append(
                (
                    names.get(c, str(c)),
                    int(metrics.nt_per_image[c]),
                    int(metrics.nt_per_class[c]),
                    *metrics.class_result(i),
                )
            )

    headers = ["Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95"]
    formatted_rows = []
    for r in rows:
        formatted_rows.append(
            [
                str(r[0]),
                str(r[1]),
                str(r[2]),
                f"{r[3]:.{decimals}f}",
                f"{r[4]:.{decimals}f}",
                f"{r[5]:.{decimals}f}",
                f"{r[6]:.{decimals}f}",
            ]
        )
    cols = list(zip(*([headers] + formatted_rows)))
    widths = [max(len(x) for x in col) for col in cols]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    for r in formatted_rows:
        lines.append("  ".join(r[i].rjust(widths[i]) if i > 0 else r[i].ljust(widths[i]) for i in range(len(r))))
    return "\n".join(lines)


def _format_device_line(device):
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        mem = int(props.total_memory / (1024 * 1024))
        return f"CUDA:{idx} ({props.name}, {mem}MiB)"
    return str(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["rgb", "ir", "dual"], required=True)
    parser.add_argument("--label", choices=["hbb", "obb"], required=True)
    parser.add_argument("--data_root", choices=["linux", "win"], default="win")
    parser.add_argument("--dataset", choices=["VEDAI", "DroneVehicle", "RGBT-Tiny"], default="VEDAI")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu", choices=["0", "1"], default="0")
    parser.add_argument(
        "--dual_mode",
        choices=[
            "shared_cat",
            "shared_catconv",
            "shared_catreduce",
            "noshare_cat",
            "noshare_catconv",
            "noshare_catreduce",
        ],
        default="shared_cat",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()


    dataset_roots_linux = {
        "VEDAI": "/home/sunyi/dataset/VEDAI/yaml",
        "DroneVehicle": "/home/sunyi/dataset/DroneVehicle/yaml",
        "RGBT-Tiny": "/home/sunyi/dataset/RGBT_Tiny/yaml",
    }
    dataset_roots_win = {
        "VEDAI": "D:/data/dataset/VEDAI/VEDAI/yaml",
        "DroneVehicle": "D:/data/dataset/DroneVehicle/DroneVehicle/yaml",
        "RGBT-Tiny": "D:/data/dataset/RGBT-Tiny/RGBT_Tiny/yaml",
    }
    data_root = dataset_roots_linux[args.dataset] if args.data_root == "linux" else dataset_roots_win[args.dataset]
    prefix = "VEDAI" if args.dataset == "VEDAI" else "DroneVehicle" if args.dataset == "DroneVehicle" else "RGBT_Tiny"
    data_map = {
        "rgb_hbb": f"{data_root}/{prefix}_rgb.yaml",
        "rgb_obb": f"{data_root}/{prefix}_rgb_obb.yaml",
        "ir_hbb": f"{data_root}/{prefix}_ir.yaml",
        "ir_obb": f"{data_root}/{prefix}_ir_obb.yaml",
        "dual_hbb": f"{data_root}/{prefix}_dual.yaml",
        "dual_obb": f"{data_root}/{prefix}_dual_obb.yaml",
    }
    key = f"{args.view}_{args.label}"
    if args.dataset == "RGBT-Tiny" and args.label == "obb":
        raise ValueError("RGBT-Tiny has no OBB labels; use --label hbb.")
    data = data_map[key]
    if args.view == "dual":
        dual_map = {
            "shared_cat": (
                "ultralytics/cfg/models/11/yolo11-obb-dual.yaml",
                "ultralytics/cfg/models/11/yolo11-dual.yaml",
            ),
            "shared_catconv": (
                "ultralytics/cfg/models/11/yolo11-obb-dual-catconv.yaml",
                "ultralytics/cfg/models/11/yolo11-dual-catconv.yaml",
            ),
            "shared_catreduce": (
                "ultralytics/cfg/models/11/yolo11-obb-dual-catreduce.yaml",
                "ultralytics/cfg/models/11/yolo11-dual-catreduce.yaml",
            ),
            "noshare_cat": (
                "ultralytics/cfg/models/11/yolo11-obb-dual-noshare.yaml",
                "ultralytics/cfg/models/11/yolo11-dual-noshare.yaml",
            ),
            "noshare_catconv": (
                "ultralytics/cfg/models/11/yolo11-obb-dual-noshare-catconv.yaml",
                "ultralytics/cfg/models/11/yolo11-dual-noshare-catconv.yaml",
            ),
            "noshare_catreduce": (
                "ultralytics/cfg/models/11/yolo11-obb-dual-noshare-catreduce.yaml",
                "ultralytics/cfg/models/11/yolo11-dual-noshare-catreduce.yaml",
            ),
        }
        model_path = dual_map[args.dual_mode][0 if args.label == "obb" else 1]
    else:
        model_path = (
            "ultralytics/cfg/models/11/yolo11-obb.yaml"
            if args.label == "obb"
            else "ultralytics/cfg/models/11/yolo11.yaml"
        )

    pretrained_path = None
    if args.pretrained:
        weights_dir = Path(__file__).resolve().parent / "pretrain_weights"
        weights_name = "yolo11n-obb.pt" if args.label == "obb" else "yolo11n.pt"
        pretrained_path = weights_dir / weights_name
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

    model = YOLO(model_path)
    run_info = {"start_time": None, "args": vars(args), "run_type": "train"}

    def on_train_start(trainer):
        run_info["start_time"] = time.time()

    def on_pretrain_routine_end(trainer):
        if not pretrained_path:
            return
        if args.view != "dual" or "noshare" not in args.dual_mode:
            return
        m = trainer.model
        if not hasattr(m, "backbone_ir") or not hasattr(m, "backbone_len"):
            return
        rgb_backbone = torch.nn.Sequential(*m.model[: m.backbone_len])
        m.backbone_ir.load_state_dict(rgb_backbone.state_dict(), strict=True)

    def on_train_end(trainer):
        save_dir = Path(trainer.save_dir)
        start_time = run_info.get("start_time") or time.time()
        end_time = time.time()
        elapsed = end_time - start_time
        validator = trainer.validator
        seen = getattr(validator, "seen", None)
        metrics = trainer.metrics
        if metrics is None or not hasattr(metrics, "mean_results"):
            metrics = getattr(validator, "metrics", metrics)
        best = trainer.best if trainer.best is not None else ""
        device_line = _format_device_line(trainer.device)
        metrics_table = _format_metrics_table(metrics, seen)
        lines = [
            f"Run type: {run_info['run_type']}",
            f"Start time: {datetime.fromtimestamp(start_time).isoformat(timespec='seconds')}",
            f"End time: {datetime.fromtimestamp(end_time).isoformat(timespec='seconds')}",
            f"Duration: {elapsed:.1f}s",
            "Args:",
        ]
        for k, v in run_info["args"].items():
            lines.append(f"  {k}: {v}")
        lines.extend(
            [
                "Validation:",
                f"Validating {best}..." if best else "Validating...",
                f"Ultralytics {ultralytics_version} Python-{platform.python_version()} torch-{torch.__version__} {device_line}",
                metrics_table,
            ]
        )
        save_dir.joinpath("desc.txt").write_text("\n".join(lines), encoding="utf-8")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_end", on_train_end)
    model.train(
        data=data,
        pretrained=str(pretrained_path) if pretrained_path else False,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.gpu,
        name=key,
        cache=False,
    )


if __name__ == "__main__":
    main()
