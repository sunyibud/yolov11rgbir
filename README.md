# Ultralytics Dual-Branch (RGB/IR) Experiments

This repo is a customized Ultralytics YOLO11 setup for RGB/IR single-branch and dual-branch experiments, with OBB
support and utility scripts for evaluation and export.

## Requirements
- Python 3.10+ recommended
- PyTorch with CUDA (if using GPU)
- Install dependencies from `pyproject.toml` (or your existing environment)

## Datasets
Supported datasets:
- VEDAI
- DroneVehicle
- RGBT-Tiny (no OBB labels)

### Directory layout (example)
```
DATASET_ROOT/
├── rgb/
│   ├── images/
│   ├── labels/        # HBB labels
│   └── obb_labels/    # OBB labels
├── ir/
│   ├── images/
│   ├── labels/
│   └── obb_labels/
├── split/
│   ├── train_default.txt
│   ├── val_default.txt
│   └── test_default.txt
└── yaml/
    ├── DATASET_rgb.yaml
    ├── DATASET_ir.yaml
    ├── DATASET_dual.yaml
    ├── DATASET_rgb_obb.yaml
    ├── DATASET_ir_obb.yaml
    └── DATASET_dual_obb.yaml
```

### Dual YAML (key fields)
```
path: /path/to/DATASET_ROOT
view: dual
label: hbb|obb
label_view: ir        # use IR labels as GT
rgb_dir: rgb/images
ir_dir: ir/images
train: split/train_default.txt
val: split/val_default.txt
test: split/test_default.txt
train_rgb: split/train_default.txt
train_ir: split/train_default.txt
val_rgb: split/val_default.txt
val_ir: split/val_default.txt
test_rgb: split/test_default.txt
test_ir: split/test_default.txt
```

## Training (train.py)
```
python train.py --view dual --label obb --dataset DroneVehicle --data_root linux --dual_mode noshare_cat --epochs 100
```

### Key arguments
- `--view`: `rgb` | `ir` | `dual`
- `--label`: `hbb` | `obb`
- `--dataset`: `VEDAI` | `DroneVehicle` | `RGBT-Tiny`
- `--data_root`: `win` | `linux` (selects dataset yaml root)
- `--dual_mode`: fusion/backbone mode for dual only
  - `shared_cat`, `shared_catconv`, `shared_catreduce`
  - `noshare_cat`, `noshare_catconv`, `noshare_catreduce`

### dual_mode details
- `shared_cat`: shared backbone for RGB/IR, concat features (2C) before neck/head.
- `shared_catconv`: shared backbone, concat then 1x1 conv (2C -> 2C) for channel mixing.
- `shared_catreduce`: shared backbone, concat then 1x1 conv reduce (2C -> C).
- `noshare_cat`: separate RGB/IR backbones, concat features (2C).
- `noshare_catconv`: separate backbones, concat then 1x1 conv (2C -> 2C).
- `noshare_catreduce`: separate backbones, concat then 1x1 conv reduce (2C -> C).
- `--pretrained`: enable pretrained weights from `pretrain_weights/`
  - uses `yolo11n.pt` or `yolo11n-obb.pt` based on `--label`
  - for dual + noshare, IR backbone copies RGB backbone weights
- `--gpu`: `0` | `1`
- `--epochs`, `--imgsz`, `--batch`, `--workers`

### Output
Runs are saved under `runs/` with `detect/` or `obb/` subfolders.
Each run writes `desc.txt` summarizing args and final validation table.

## Testing (test.py)
```
python test.py --model runs/obb/dual_obb/weights/best.pt --view dual --label obb --dataset DroneVehicle --split test
```

## DOTA Export (eval.py)
Export predictions for DOTA_devkit evaluation:
```
python eval.py --model runs/obb/dual_obb/weights/best.pt --view dual --label obb \
  --dataset DroneVehicle --split test --out_dir eval/DroneVehicle_dual_obb
```
Outputs one text file per class:
```
image_id score x1 y1 x2 y2 x3 y3 x4 y4
```

## Notes
- Dual mode uses RGB+IR inputs (6 channels). Ensure `train_rgb/val_rgb/train_ir/val_ir` are set in dual yaml.
- RGBT-Tiny has no OBB labels; use `--label hbb`.
