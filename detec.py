import cv2
from ultralytics import YOLO

cctv = "rtsp://192.168.60.100:554/user=Panjikrisnanda_password=XZlsoHxN_channel=0_stream=0&onvif=0.sdp?real_stream"

model = YOLO("yolo11s.pt")
kamera = cv2.VideoCapture(cctv, cv2.CAP_FFMPEG)

while True:
    ret, frame = kamera.read()
    #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = cv2.resize(frame, (640, 360))


    result = model.predict(frame, classes=[0], imgsz=960, conf=0.5, verbose=False)
    annotated_frame = result[0].plot()

    resize_win = cv2.resize(annotated_frame, (800, 600))
    cv2.imshow("Detect!", resize_win)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break