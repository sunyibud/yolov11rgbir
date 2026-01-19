# Ultralytics 双分支（RGB/IR）实验说明

本项目是在 Ultralytics YOLO11 基础上改造的 RGB/IR 单分支与双分支实验框架，支持 OBB，并提供训练、测试与
DOTA 评估导出脚本。

## 环境说明
- 建议 Python 3.10+
- 使用带 CUDA 的 PyTorch
- 依赖可参考 `pyproject.toml`（或你已有环境）

## 数据集
已适配的数据集：
- VEDAI
- DroneVehicle
- RGBT-Tiny（无 OBB 标签）

### 目录结构示例
```
DATASET_ROOT/
├── rgb/
│   ├── images/
│   ├── labels/        # HBB 标签
│   └── obb_labels/    # OBB 标签
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

### 双分支 YAML 关键字段
```
path: /path/to/DATASET_ROOT
view: dual
label: hbb|obb
label_view: ir        # 使用 IR 标签作为 GT
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

## 训练（train.py）
```
python train.py --view dual --label obb --dataset DroneVehicle --data_root linux --dual_mode noshare_cat --epochs 100
```

### 常用参数
- `--view`: `rgb` | `ir` | `dual`
- `--label`: `hbb` | `obb`
- `--dataset`: `VEDAI` | `DroneVehicle` | `RGBT-Tiny`
- `--data_root`: `win` | `linux`（选择数据集 YAML 根目录）
- `--dual_mode`: 仅双分支使用
  - `shared_cat`, `shared_catconv`, `shared_catreduce`
  - `noshare_cat`, `noshare_catconv`, `noshare_catreduce`

### dual_mode 说明
- `shared_cat`：RGB/IR 共享 backbone，特征 concat 后送入 neck/head（2C）。
- `shared_catconv`：共享 backbone，concat 后接 1x1 conv 做通道混合（2C -> 2C）。
- `shared_catreduce`：共享 backbone，concat 后接 1x1 conv 降维（2C -> C）。
- `noshare_cat`：RGB/IR 独立 backbone，特征 concat（2C）。
- `noshare_catconv`：独立 backbone，concat 后接 1x1 conv（2C -> 2C）。
- `noshare_catreduce`：独立 backbone，concat 后接 1x1 conv 降维（2C -> C）。
- `--pretrained`: 使用 `pretrain_weights/` 下的预训练权重
  - `hbb` -> `yolo11n.pt`
  - `obb` -> `yolo11n-obb.pt`
  - 双分支 + noshare 时会把 RGB backbone 权重复制给 IR backbone
- `--gpu`: `0` | `1`
- `--epochs`, `--imgsz`, `--batch`, `--workers`

### 输出
日志保存在 `runs/` 下的 `detect/` 或 `obb/` 子目录，运行结束会生成 `desc.txt`，记录参数和最终指标表。

## 测试（test.py）
```
python test.py --model runs/obb/dual_obb/weights/best.pt --view dual --label obb --dataset DroneVehicle --split test
```

## DOTA 导出（eval.py）
用于 DOTA_devkit 评估的预测结果导出：
```
python eval.py --model runs/obb/dual_obb/weights/best.pt --view dual --label obb \
  --dataset DroneVehicle --split test --out_dir eval/DroneVehicle_dual_obb
```
输出为每个类别一个 txt：
```
image_id score x1 y1 x2 y2 x3 y3 x4 y4
```

## 备注
- 双分支一定要在 YAML 中显式写出 `train_rgb/val_rgb/train_ir/val_ir`，确保 IR 读取正确。
- RGBT-Tiny 没有 OBB 标签，训练/测试请使用 `--label hbb`。
