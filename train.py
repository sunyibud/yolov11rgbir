from ultralytics import YOLO

def main():
    model = YOLO(r'D:\projects\python\ultralytics\ultralytics\cfg\models\11\yolo11l.yaml')
    # model = YOLO(r'D:\projects\python\ultralytics\ultralytics\cfg\models\transformer\yolov11_fusion_transformerx3_FLIR_aligned.yaml')
    # model.load(r'D:\projectdata\python\postgraduate\ultralytics\output\exp2\weights\best.pt')
    # model = YOLO(r'D:\projects\python\ultralytics\output\exp5\weights\best.pt')
    model.train(
        data=r'/usr/local/sunyi/projects/ultralytics/train_data/LL/data.yaml',
        imgsz=640,
        epochs=10,
        batch=16,
        close_mosaic=False,  #Mosaic 数据增强是在训练过程中将 4 张随机选择的图片拼接成一张图片
        workers=8,
        optimizer='SGD',
        project='./output',
        name='exp',
        device=0,
    )
    # metrics = model.val(
    #     data=r'D:\projectdata\python\postgraduate\ultralytics\train_data\mask\data.yaml',
    #     imgsz=640,
    #     batch=32,
    #     conf=0.25,
    #     iou=0.6,
    #     project='./output',
    #     name='exp',
    #     device='0',
    # )
    # images = [r'D:\projects\python\ultralytics\test_image\img_5.png']
    # results = model(images)   # return a list of Results objects
    # Process results list
    # for ind, result in enumerate(results):
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     result.show()  # display to screen
        # result.save(filename="./result/"+images[ind])  # save to disk

if __name__ == '__main__':
    main()
