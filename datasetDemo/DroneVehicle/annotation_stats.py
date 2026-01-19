import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ================= 配置区域 =================
# 脚本运行在数据集根目录下
ROOT_DIR = "." 

# 定义 Split 文件名
SPLITS = ['train', 'val', 'test']

# 类别名称 (必须与之前的映射顺序一致)
CLASS_NAMES = [
    'car',          # 0
    'truck',        # 1
    'bus',          # 2
    'van',          # 3
    'freight_car'   # 4
]

# 标签子目录名称 (这里默认统计 OBB 标签，如果您想统计 HBB，改为 'labels')
LABEL_SUBDIR = "obb_labels" 
# ===========================================

def parse_txt_counts(txt_file, counter_dict):
    """解析 YOLO 格式 txt 文件统计类别"""
    if not os.path.exists(txt_file):
        return
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                # 获取 class_id
                try:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(CLASS_NAMES):
                        name = CLASS_NAMES[class_id]
                        counter_dict[name] += 1
                except ValueError:
                    pass
    except Exception:
        pass

def main():
    print(f"📊 开始统计数据集: {os.path.abspath(ROOT_DIR)}")
    print(f"   统计标签类型: {LABEL_SUBDIR}")
    
    # 统计容器
    stats = {
        'rgb': defaultdict(int),
        'ir': defaultdict(int)
    }
    
    img_counts = {
        'rgb': 0,
        'ir': 0,
        'aligned': 0
    }
    
    split_info = {}

    # 遍历 train/val/test
    for split in SPLITS:
        split_txt_path = os.path.join(ROOT_DIR, "split", f"{split}.txt")
        
        if not os.path.exists(split_txt_path):
            print(f"⚠️ 警告: 找不到 split 文件 {split_txt_path}")
            continue
            
        # 读取该 split 下的所有图片名
        with open(split_txt_path, 'r') as f:
            file_names = [line.strip() for line in f.readlines() if line.strip()]
            
        split_info[split] = len(file_names)
        print(f"📦 正在扫描 {split} 集 ({len(file_names)} 张)...")
        
        for fname in tqdm(file_names):
            # 推导 ID 和 TXT 文件名
            # fname 类似 "00001.jpg"
            file_id = os.path.splitext(fname)[0]
            txt_name = file_id + ".txt"
            
            # 构建路径
            # RGB
            p_rgb_img = os.path.join(ROOT_DIR, "rgb", "images", fname)
            p_rgb_lbl = os.path.join(ROOT_DIR, "rgb", LABEL_SUBDIR, txt_name)
            
            # IR
            p_ir_img = os.path.join(ROOT_DIR, "ir", "images", fname)
            p_ir_lbl = os.path.join(ROOT_DIR, "ir", LABEL_SUBDIR, txt_name)
            
            # 1. 检查图片存在性
            has_rgb = os.path.exists(p_rgb_img)
            has_ir = os.path.exists(p_ir_img)
            
            if has_rgb: img_counts['rgb'] += 1
            if has_ir: img_counts['ir'] += 1
            if has_rgb and has_ir: img_counts['aligned'] += 1
            
            # 2. 统计标签
            if has_rgb:
                parse_txt_counts(p_rgb_lbl, stats['rgb'])
            if has_ir:
                parse_txt_counts(p_ir_lbl, stats['ir'])

    # --- 生成统计表格 ---
    data = []
    # 保证顺序
    for cat in CLASS_NAMES:
        data.append({
            'Category': cat,
            'RGB': stats['rgb'][cat],
            'Infrared': stats['ir'][cat]
        })
    
    # 添加总计
    data.append({
        'Category': 'TOTAL',
        'RGB': sum(stats['rgb'].values()),
        'Infrared': sum(stats['ir'].values())
    })

    df = pd.DataFrame(data)
    # 转置表格
    df_transposed = df.set_index('Category').T
    
    print("\n" + "="*60)
    print(f"FINAL STATISTICS (Based on {LABEL_SUBDIR})")
    print("="*60)
    print(df_transposed)
    print("="*60)
    
    print("\n" + "="*60)
    print("DATASET STRUCTURE SUMMARY")
    print("="*60)
    print(f"Total RGB Images      : {img_counts['rgb']}")
    print(f"Total Infrared Images : {img_counts['ir']}")
    print(f"Aligned Pairs         : {img_counts['aligned']}")
    print("-" * 30)
    for split, count in split_info.items():
        print(f"Split '{split:<5}' : {count} images")
    print("="*60)

if __name__ == "__main__":
    main()