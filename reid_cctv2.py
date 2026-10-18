import cv2
import os
import time
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO

cctv = "rtsp://192.168.60.100:554/user=Panjikrisnanda_password=XZlsoHxN_channel=0_stream=0&onvif=0.sdp?real_stream"

model = YOLO("yolo11s.pt")
face_model = YOLO("yolo11n-face.pt")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=-1, det_size=(320, 320))

training = cv2.imread("./training/yuda.jpeg")

if training is None:
    print("Foto training tidak ditemukan!")
    exit()

training_face = face_app.get(training)

if len(training_face) == 0:
    print("Wajah training tidak ditemukan!")
    exit()

training_ncode = training_face[0].normed_embedding

kamera = cv2.VideoCapture(cctv, cv2.CAP_FFMPEG)
kamera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

os.makedirs("hasil/body", exist_ok=True)
os.makedirs("hasil/face", exist_ok=True)

nomor = 0
waktu_simpan = 0
hitung = 0

while True:
    ret, frame = kamera.read()

    if not ret or frame is None:
        print("Frame gagal dibaca, mencoba lagi...")
        continue

    # Kecilkan frame untuk mempercepat proses
    frame = cv2.resize(frame, (960, 540))

    hitung += 1

    # Deteksi orang
    result = model.predict(
        frame,
        classes=[0],
        imgsz=640,
        conf=0.5,
        verbose=False
    )

    for box in result[0].boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        body = frame[y1:y2, x1:x2]

        if body.size == 0:
            continue

        nama = "Mencari wajah..."

        # Deteksi wajah dan ArcFace setiap 5 frame
        if hitung % 5 == 0:
            wajah = face_model.predict(
                body,
                imgsz=320,
                conf=0.25,
                verbose=False
            )

            for face_box in wajah[0].boxes:
                fx1, fy1, fx2, fy2 = map(
                    int, face_box.xyxy[0].tolist()
                )

                lebar = fx2 - fx1
                tinggi = fy2 - fy1

                # Tambahkan margin crop wajah
                cx1 = max(0, fx1 - int(lebar * 0.2))
                cy1 = max(0, fy1 - int(tinggi * 0.2))
                cx2 = min(body.shape[1], fx2 + int(lebar * 0.2))
                cy2 = min(body.shape[0], fy2 + int(tinggi * 0.2))

                face = body[cy1:cy2, cx1:cx2]

                if face.size == 0:
                    continue

                # ArcFace
                wajah_arc = face_app.get(face)

                if len(wajah_arc) > 0:
                    ncode = wajah_arc[0].normed_embedding

                    compare = np.dot(training_ncode, ncode)

                    print("Similarity:", round(float(compare), 3))

                    if compare > 0.4:
                        nama = "Yuda"
                    else:
                        nama = "Unknown"

                    # Gambar kotak wajah
                    cv2.rectangle(
                        frame,
                        (x1 + fx1, y1 + fy1),
                        (x1 + fx2, y1 + fy2),
                        (255, 0, 0),
                        2
                    )

        # Gambar kotak badan
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Tampilkan nama
        cv2.putText(
            frame,
            nama,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Tampilkan frame
    resize_win = cv2.resize(frame, (800, 450))
    cv2.imshow("Detect!", resize_win)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
