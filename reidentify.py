import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
kamera = cv2.VideoCapture(0)
track_id = []

while True:
    ret, frame = kamera.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = model.track(frame, persist=True, tracker= "botsort_reid.yaml", classes=[0], conf=0.5, verbose=False)
    an_frame = result[0].plot()

    for box in result[0].boxes:
        atas, kanan, bawah, kiri = map(int, box.xyxy[0].tolist())
        id = int(box.id[0])
        crop = frame[kanan:kiri, atas:bawah]

        if crop.size > 0:
            cv2.imshow("Crop", crop)
            cv2.imwrite(f"./img/{id}.jpg", crop)
            if id not in track_id:
                track_id.append(id)


    cv2.imshow("Detect!", an_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(track_id)
        break

