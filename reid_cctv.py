
import cv2
import os
import time
import face_recognition
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO

cctv = "rtsp://192.168.60.100:554/user=Panjikrisnanda_password=XZlsoHxN_channel=0_stream=0&onvif=0.sdp?real_stream"

model = YOLO("yolo11s.pt")
face_model = YOLO("yolo11n-face.pt")

# training = face_recognition.load_image_file("./training/yuda.jpeg")
# training_ncode = face_recognition.face_encodings(training)

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

training = cv2.imread("./training/yuda.jpeg")
training_face = face_app.get(training)
training_ncode = training_face[0].normed_embedding

kamera = cv2.VideoCapture(cctv, cv2.CAP_FFMPEG)

os.makedirs("hasil/body", exist_ok=True)
os.makedirs("hasil/face", exist_ok=True)

nomor = 0
waktu_simpan = 0

while True:
    ret, frame = kamera.read()

    if not ret or frame is None:
        print("Frame gagal dibaca, mencoba lagi...")
        continue
    
    result = model.predict(
        frame,
        classes=[0],
        imgsz=960,
        conf=0.5,
        verbose=False
    )

    for box in result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        body = frame[y1:y2, x1:x2]

        if body.size == 0:
            continue

        # Deteksi wajah dari crop badan
        wajah = face_model.predict(
            body,
            conf=0.25,
            verbose=False
        )

        for face_box in wajah[0].boxes:
            fx1, fy1, fx2, fy2 = map(
                int, face_box.xyxy[0].tolist()
            )

            lebar = fx2 - fx1
            tinggi = fy2 - fy1

            fx1 = max(0, fx1 - int(lebar * 0.2))
            fy1 = max(0, fy1 - int(tinggi * 0.2))
            fx2 = min(body.shape[1], fx2 + int(lebar * 0.2))
            fy2 = min(body.shape[0], fy2 + int(tinggi * 0.2))

            face = body[fy1:fy2, fx1:fx2]

            if face.size == 0:
                continue

            grey = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            grey2 = face_app.get(grey)
            # ncode = face_recognition.face_encodings(grey)

            if len(grey2) > 0:
                # compare = face_recognition.face_distance(training_ncode, ncode[0])
                ncode = grey2[0].normed_embedding
                compare = np.dot(training_ncode, ncode)
                print(compare)
            else:
                print("no face detected")

        #     # Simpan crop setiap 3 detik
        #     if time.time() - waktu_simpan >= 3:
        #         nomor += 1

        #         cv2.imwrite(
        #             f"hasil/body/body_{nomor}.jpg",
        #             body
        #         )

        #         cv2.imwrite(
        #             f"hasil/face/face_{nomor}.jpg",
        #             face
        #         )

        #         waktu_simpan = time.time()
        #         print("Crop tersimpan:", nomor)

        # Tampilkan kotak badan
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        # Tampilkan kotak wajah
        for face_box in wajah[0].boxes:
            fx1, fy1, fx2, fy2 = map(
                int, face_box.xyxy[0].tolist()
            )

            cv2.rectangle(
                frame,
                (x1 + fx1, y1 + fy1),
                (x1 + fx2, y1 + fy2),
                (255, 0, 0), 2
            )

    an_frame = result[0].plot()
    resize_win = cv2.resize(an_frame, (800, 600))

    cv2.imshow("Detect!", resize_win)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
