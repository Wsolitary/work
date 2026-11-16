from ultralytics import YOLO

def main():
    #加载模型
    model_path = "D:/RM/yolo/YOLO/runs/detect/rm_test/yolov8n_50ep/weights/best.pt"
    model = YOLO(model_path)
    img_path = "D:/RM/yolo/YOLO/test/test1.png"
    # conf=0.25 表示只显示置信度大于 25% 的框
    results = model.predict(source=img_path, save=True, conf=0.25)
    print("测试完成！结果位于'runs/detect/predict' 文件夹下。")

if __name__ == '__main__':
    main()
