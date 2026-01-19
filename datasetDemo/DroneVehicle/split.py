import os
import random
import glob
import math

# ================= 配置区域 =================
# 数据集根目录 (脚本应放在 DroneVehicle 根目录下)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 图片所在路径 (以 RGB 图片为基准)
IMG_DIR = os.path.join(BASE_DIR, "rgb", "images")
# 输出 Split 文件的目录
SPLIT_DIR = os.path.join(BASE_DIR, "split")

# 划分比例 [Train, Val, Test]
# 确保加起来等于 1.0
RATIOS = [0.7, 0.2, 0.1] 

# 随机种子 (修改此数可改变划分结果，固定此数可复现结果)
RANDOM_SEED = 42

# 图片后缀过滤
IMG_EXTS = ['.jpg', '.png', '.jpeg', '.bmp']
# ===========================================

def main():
    print(f"🚀 开始重新划分数据集: {BASE_DIR}")
    print(f"🎲 随机种子: {RANDOM_SEED}")
    print(f"📊 划分比例: 训练={RATIOS[0]*100}% | 验证={RATIOS[1]*100}% | 测试={RATIOS[2]*100}%")

    # 1. 检查目录
    if not os.path.exists(IMG_DIR):
        print(f"❌ 错误: 找不到图片目录 {IMG_DIR}")
        return
    
    if not os.path.exists(SPLIT_DIR):
        os.makedirs(SPLIT_DIR)

    # 2. 获取所有图片文件名
    all_files = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    total_count = len(all_files)
    
    if total_count == 0:
        print("❌ 目录下没有找到图片。")
        return

    print(f"📂 扫描到图片总数: {total_count}")

    # 3. 随机打乱
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    # 4. 计算切分点
    n_train = int(total_count * RATIOS[0])
    n_val = int(total_count * RATIOS[1])
    # 剩下的全部给 test，确保总数对齐
    n_test = total_count - n_train - n_val

    # 5. 切分列表
    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val :]

    # 6. 写入 txt 文件
    def write_split(filename, file_list):
        path = os.path.join(SPLIT_DIR, filename)
        # 排序后再写入，看着整齐（虽然内容是随机抽取的）
        file_list.sort() 
        with open(path, 'w') as f:
            for name in file_list:
                f.write(name + "\n")
        return path

    p1 = write_split("train.txt", train_files)
    p2 = write_split("val.txt", val_files)
    p3 = write_split("test.txt", test_files)

    print("\n" + "="*40)
    print("✅ 划分完成！文件已覆盖生成:")
    print(f"  📄 train.txt: {len(train_files)} 张 ({len(train_files)/total_count:.1%})")
    print(f"  📄 val.txt:   {len(val_files)} 张 ({len(val_files)/total_count:.1%})")
    print(f"  📄 test.txt:  {len(test_files)} 张 ({len(test_files)/total_count:.1%})")
    print("="*40)
    print(f"输出目录: {SPLIT_DIR}")

if __name__ == "__main__":
    main()