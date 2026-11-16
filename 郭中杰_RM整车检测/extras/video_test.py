import cv2
from ultralytics import YOLO

def main():
    model = YOLO("D:/RM/yolo/YOLO/runs/detect/rm_test/yolov8n_50ep2/weights/best.pt") 
    video_path = "D:/RM/yolo/YOLO/test/my_video.mp4" 
    cap = cv2.VideoCapture(video_path)
    #从视频的第420秒开始处理
    start_time_sec = 420
    start_time_msec = start_time_sec * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec) 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
    print(f"正在从第 {start_time_sec} 秒开始处理视频...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("处理完毕！")#其实根本不会看完整个视频
            break
            
        results = model.predict(source=frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()
        
        cv2.imshow("RM Video Test (Press Q to quit)", annotated_frame)
        out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()