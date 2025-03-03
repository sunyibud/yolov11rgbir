# diy训练过程
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)
'''
args: 训练参数
'''
def train(args):
    model_yaml, data_yaml, imgsz, epochs, batch, close_mosaic, workers, optimizer, project, name, device = args.values()
    data_dict = yaml_load(data_yaml)
    train_data_rgb_path, train_data_ir_path, val_data_rgb_path, val_data_ir_path = \
        data_dict['train_rgb'], data_dict['train_ir'], data_dict['val_rgb'], data_dict['val_ir']
    # try:
    #     model = YOLO(model_yaml)
    #     print("YOLO 初始化成功！")
    # except Exception as e:
    #     print(f"YOLO 初始化失败，错误信息：{e}")
    model = YOLO(model_yaml)
    model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        close_mosaic=close_mosaic,  # Mosaic 数据增强是在训练过程中将 4 张随机选择的图片拼接成一张图片
        workers=workers,
        optimizer=optimizer,
        project=project,
        name=name,
        device=device,
    )


def main():
    args = dict(
        model_yaml = r'/sunyi/projects/yolov11rgbir/ultralytics/cfg/models/transformer/yolov11_fusion_transformerx3_FLIR_aligned.yaml',
        data_yaml=r'/sunyi/projects/yolov11rgbir/train_data/LLVIP/data.yaml',
        # model_yaml=r'E:\projects\python\ultralytics\ultralytics\cfg\models\transformer\yolov11_fusion_transformerx3_FLIR_aligned.yaml',
        # data_yaml=r'E:\projects\python\ultralytics\train_data\LLVIP\data.yaml',
        imgsz=640,
        epochs=10,
        batch=16,
        close_mosaic=False,  # Mosaic 数据增强是在训练过程中将 4 张随机选择的图片拼接成一张图片
        workers=8,
        optimizer='SGD',
        project='./output',
        name='exp',
        device=0,
    )
    train(args)

if __name__ == '__main__':
    main()