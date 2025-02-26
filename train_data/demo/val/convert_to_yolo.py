import os
import json

# COCO标注文件路径
json_file_path = './thermal_annotations.json'
# 图像文件夹路径
images_folder_path = './images'
# YOLO标签保存路径
yolo_labels_path = './yolo_labels'
os.makedirs(yolo_labels_path, exist_ok=True)

# 读取COCO标注文件
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 类别映射，从json中读取
category_mapping = {category['id']: category['name'] for category in data['categories']}

# 创建图像ID到文件名的映射
image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}
# 创建图像ID到图像尺寸的映射
image_id_to_size = {image['id']: (image['width'], image['height']) for image in data['images']}
# 处理每个标注
for annotation in data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']  # COCO格式的bbox: [x_min, y_min, width, height]

    # 获取对应的图像文件名
    image_filename = image_id_to_filename[image_id]
    image_path = os.path.join(images_folder_path, image_filename)

    # # 检查图像文件是否存在
    # if not os.path.exists(image_path):
    #     print(f"图像文件 {image_path} 不存在，跳过该标注。")
    #     continue

    # 获取图像尺寸
    # with open(image_path, 'rb') as img_file:
    #     from PIL import Image
    #     img = Image.open(img_file)
    #     image_width, image_height = img.size
    image_width, image_height = image_id_to_size[image_id]

    # 计算YOLO格式的bbox中心点坐标和宽高的相对值
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    width /= image_width
    height /= image_height

    # 获取类别索引
    if category_id not in category_mapping:
        print(f"类别ID {category_id} 未在类别映射中定义，跳过该标注。")
        continue

    # 准备YOLO格式的标注内容
    yolo_annotation = f"{category_id} {x_center} {y_center} {width} {height}\n"

    # 保存YOLO格式的标注到对应的txt文件
    txt_filename = os.path.splitext(image_filename)[0] + '.txt'
    txt_file_path = os.path.join(yolo_labels_path, txt_filename)
    with open(txt_file_path, 'a') as txt_file:
        txt_file.write(yolo_annotation)
