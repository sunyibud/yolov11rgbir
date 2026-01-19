import os
import glob
from collections import Counter
from tqdm import tqdm

# ================= 配置区域 =================
# 脚本所在目录 (即 DroneVehicle 根目录)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 类别映射
CLASS_MAP = {
    0: "Car",
    1: "Truck",
    2: "Bus",
    3: "Van",
    4: "Freight_Car"
}

# 子集列表
SPLITS = ["train", "val", "test"]

# 标签路径配置 (基于 RGB 统计即可，因为我们已经做过双光对齐)
LABEL_DIRS = {
    "HBB (水平框)": os.path.join(BASE_DIR, "rgb", "labels"),
    "OBB (旋转框)": os.path.join(BASE_DIR, "rgb", "obb_labels")
}
# ===========================================

def get_ids_from_split(split_name):
    """读取 split/xxx.txt 获取文件名列表"""
    txt_path = os.path.join(BASE_DIR, "split", f"{split_name}.txt")
    if not os.path.exists(txt_path):
        print(f"⚠️ 找不到划分文件: {txt_path}")
        return []
    
    with open(txt_path, 'r') as f:
        # 文件内容如 00001.jpg，我们需要去掉后缀拿到 ID 00001
        ids = [os.path.splitext(line.strip())[0] for line in f if line.strip()]
    return ids

def print_table(split_name, task_name, counter, total_objs, total_imgs):
    print(f"\n>>> 子集: {split_name.upper()} | 任务: {task_name}")
    print(f"    图片数: {total_imgs} | 目标总数: {total_objs}")
    print("-" * 65)
    print(f"{'ID':<4} | {'类别名称':<12} | {'数量':<8} | {'占比':<7} | {'可视化'}")
    print("-" * 65)
    
    # 按 ID 顺序输出 0-4
    for cid in range(len(CLASS_MAP)):
        name = CLASS_MAP[cid]
        count = counter[cid]
        ratio = (count / total_objs * 100) if total_objs > 0 else 0
        
        # 进度条
        bar = "█" * int(ratio / 2)
        
        # 状态标记
        note = ""
        if count == 0:
            note = "⚠️ 无样本"
        elif count < 100:
            note = "⚠️ 稀缺"
            
        print(f"{cid:<4} | {name:<12} | {count:<8} | {ratio:>6.2f}% | {bar} {note}")
    print("-" * 65)

def main():
    print(f"🚀 开始统计最终数据集: {BASE_DIR}\n")
    
    # 检查目录是否存在
    for name, path in LABEL_DIRS.items():
        if not os.path.exists(path):
            print(f"❌ 错误: 找不到标签目录 {path}")
            return

    global_hbb_count = Counter()
    global_obb_count = Counter()

    # 遍历 Train, Val, Test
    for split in SPLITS:
        ids = get_ids_from_split(split)
        if not ids:
            continue
            
        # 针对每个子集，统计 HBB 和 OBB
        for task_name, label_root in LABEL_DIRS.items():
            current_counter = Counter()
            current_total_objs = 0
            
            # 遍历该子集下的所有图片ID
            for uid in tqdm(ids, desc=f"Scanning {split} {task_name.split()[0]}"):
                txt_path = os.path.join(label_root, f"{uid}.txt")
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) > 0:
                                try:
                                    cls_id = int(parts[0])
                                    current_counter[cls_id] += 1
                                    current_total_objs += 1
                                except: pass
            
            # 打印该子集的表格
            print_table(split, task_name, current_counter, current_total_objs, len(ids))
            
            # 汇总到全局
            if "HBB" in task_name:
                global_hbb_count.update(current_counter)
            else:
                global_obb_count.update(current_counter)

    # 打印全局汇总 (可选)
    print("\n" + "="*65)
    print("🏆 全局统计汇总 (Train + Val + Test)")
    print("-" * 65)
    print(f"{'类别':<12} | {'HBB总数':<10} | {'OBB总数':<10}")
    print("-" * 65)
    for cid in range(len(CLASS_MAP)):
        name = CLASS_MAP[cid]
        h_cnt = global_hbb_count[cid]
        o_cnt = global_obb_count[cid]
        print(f"{name:<12} | {h_cnt:<10} | {o_cnt:<10}")
    print("-" * 65)
    
    # 最终确认 Freight_Car
    fc_count = global_obb_count[4]
    if fc_count > 0:
        print(f"\n✅ 成功检测到 Freight_Car (共 {fc_count} 个)！之前的拼写错误修复成功。")
    else:
        print(f"\n❌ 警告: Freight_Car 数量仍为 0，请检查之前的 XML 修正步骤。")

if __name__ == "__main__":
    main()