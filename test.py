import argparse
import platform
import time
from datetime import datetime
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained weights, e.g. runs/obb/xxx/weights/best.pt")
    parser.add_argument("--view", choices=["rgb", "ir", "dual"], required=True)
    parser.add_argument("--label", choices=["hbb", "obb"], required=True)
    parser.add_argument("--data_root", choices=["linux", "win"], default="win")
    parser.add_argument("--dataset", choices=["VEDAI", "DroneVehicle", "RGBT-Tiny"], default="VEDAI")
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
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    dataset_roots_linux = {
        "VEDAI": "/home/sunyi/dataset/VEDAI/yaml",
        "DroneVehicle": "/home/sunyi/dataset/DroneVehicle/yaml",
        "RGBT-Tiny": "/home/sunyi/dataset/RGBT_Tiny/yaml",
    }
    dataset_roots_win = {
        "VEDAI": "D:/data/dataset/VEDAI/VEDAI/yaml",
        "DroneVehicle": "D:/data/dataset/DroneVehicle/yaml",
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

    model = YOLO(args.model)
    run_info = {"start_time": time.time(), "args": vars(args), "run_type": "test"}

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

    def on_val_end(validator):
        save_dir = Path(validator.save_dir)
        start_time = run_info.get("start_time") or time.time()
        end_time = time.time()
        elapsed = end_time - start_time
        metrics = validator.metrics
        seen = getattr(validator, "seen", None)
        device_line = _format_device_line(validator.device)
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
                f"Validating {args.model}...",
                f"Ultralytics {ultralytics_version} Python-{platform.python_version()} torch-{torch.__version__} {device_line}",
                metrics_table,
            ]
        )
        save_dir.joinpath("desc.txt").write_text("\n".join(lines), encoding="utf-8")

    model.add_callback("on_val_end", on_val_end)
    metrics = model.val(
        data=data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
    )
    print(metrics)


if __name__ == "__main__":
    main()
