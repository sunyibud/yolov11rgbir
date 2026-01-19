import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import TQDM
from ultralytics.utils import YAML


def _normalize_stem(stem: str, suffixes: list[str]) -> str:
    for s in suffixes:
        if s and stem.endswith(s):
            return stem[: -len(s)]
    return stem


def _resolve_entry(entry: str, root: Path, list_dir: str | None, suffixes: list[str]) -> Path:
    entry = entry.strip()
    if not entry:
        raise FileNotFoundError("Empty entry in split file.")
    if entry.startswith("./"):
        entry = entry[2:]
    p = Path(entry)
    if p.is_absolute() and p.exists():
        return p
    if (root / entry).exists():
        return root / entry

    base = p.stem
    ext = p.suffix
    candidates = []
    if ext:
        candidates.append(entry)
    else:
        for s in suffixes:
            if s:
                candidates.append(base + s)
        candidates.append(base)

    search_root = root / list_dir if list_dir else root
    for cand in candidates:
        cand_path = Path(cand)
        if cand_path.suffix:
            path = search_root / cand_path
            if path.exists():
                return path
        else:
            for suf in IMG_FORMATS:
                path = search_root / f"{cand}.{suf}"
                if path.exists():
                    return path
    raise FileNotFoundError(f"Entry '{entry}' not found under {search_root}")


def _load_split_list(data: dict, split: str, view: str) -> list[dict]:
    root = Path(data.get("path", "")).expanduser()
    if split not in data:
        raise ValueError(f"Split '{split}' not found in yaml.")
    split_path = Path(data[split])
    if not split_path.is_absolute():
        split_path = root / split_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    lines = split_path.read_text(encoding="utf-8").strip().splitlines()
    entries = [x.strip() for x in lines if x.strip()]

    rgb_suffixes = [data.get("rgb_suffix"), "_co", "_rgb", "_vis"]
    ir_suffixes = [data.get("ir_suffix"), "_ir", "_thermal", "_tir"]

    items = []
    if view == "dual":
        rgb_dir = data.get("rgb_dir") or data.get("list_dir")
        ir_dir = data.get("ir_dir") or data.get("list_dir")
        if not rgb_dir or not ir_dir:
            raise ValueError("dual view requires rgb_dir and ir_dir (or list_dir) in yaml.")
        for e in entries:
            rgb_path = _resolve_entry(e, root, rgb_dir, rgb_suffixes)
            ir_path = _resolve_entry(e, root, ir_dir, ir_suffixes)
            image_id = _normalize_stem(rgb_path.stem, rgb_suffixes)
            items.append({"id": image_id, "rgb": rgb_path, "ir": ir_path})
    else:
        list_dir = data.get("list_dir")
        suffixes = rgb_suffixes if view == "rgb" else ir_suffixes
        for e in entries:
            img_path = _resolve_entry(e, root, list_dir, suffixes)
            image_id = _normalize_stem(img_path.stem, suffixes)
            items.append({"id": image_id, "rgb": img_path})
    return items


def _load_dual_batch(batch):
    imgs = []
    for item in batch:
        rgb = cv2.imread(str(item["rgb"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Image Not Found {item['rgb']}")
        ir = cv2.imread(str(item["ir"]), cv2.IMREAD_COLOR)
        if ir is None:
            raise FileNotFoundError(f"Image Not Found {item['ir']}")
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if ir.ndim == 2:
            ir = ir[..., None]
        img = np.concatenate((rgb, ir), axis=2)
        imgs.append(img)
    return imgs


def _xyxy_to_xyxyxyxy(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.T
    return np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained weights, e.g. runs/obb/xxx/weights/best.pt")
    parser.add_argument("--view", choices=["rgb", "ir", "dual"], required=True)
    parser.add_argument("--label", choices=["hbb", "obb"], required=True)
    parser.add_argument("--data_root", choices=["linux", "win"], default="win")
    parser.add_argument("--dataset", choices=["VEDAI", "DroneVehicle", "RGBT-Tiny"], default="VEDAI")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--out_dir", default=None)
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
    data_yaml = data_map[key]
    data = YAML.load(data_yaml)

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / "dota" / f"{args.dataset}_{key}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _load_split_list(data, args.split, args.view)
    model = YOLO(args.model)
    if args.view == "dual" and hasattr(model, "model") and hasattr(model.model, "yaml"):
        model.model.yaml["channels"] = 6

    class_names = model.names or {}
    class_files = {name: [] for name in class_names.values()}

    for i in TQDM(range(0, len(items), args.batch), desc="Exporting", total=len(range(0, len(items), args.batch))):
        batch = items[i : i + args.batch]
        if args.view == "dual":
            imgs = _load_dual_batch(batch)
            results = model.predict(
                imgs,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )
        else:
            sources = [str(x["rgb"]) for x in batch]
            results = model.predict(
                sources,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )

        for item, result in zip(batch, results):
            image_id = item["id"]
            if args.label == "obb":
                if result.obb is None:
                    continue
                xyxyxyxy = result.obb.xyxyxyxy
                conf = result.obb.conf
                cls = result.obb.cls
            else:
                if result.boxes is None:
                    continue
                xyxy = result.boxes.xyxy
                xyxyxyxy = _xyxy_to_xyxyxyxy(xyxy)
                conf = result.boxes.conf
                cls = result.boxes.cls

            xyxyxyxy = xyxyxyxy.cpu().numpy() if hasattr(xyxyxyxy, "cpu") else np.asarray(xyxyxyxy)
            conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
            cls = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)

            for box, score, c in zip(xyxyxyxy, conf, cls):
                name = class_names.get(int(c), str(int(c)))
                flat = box.reshape(-1).tolist()
                line = f"{image_id} {score:.6f} " + " ".join(f"{v:.2f}" for v in flat)
                class_files.setdefault(name, []).append(line)

    for name, lines in class_files.items():
        (out_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.time()
    print(f"Saved DOTA-format results to: {out_dir}")


if __name__ == "__main__":
    main()
