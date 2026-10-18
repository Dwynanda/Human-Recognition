 # for face_box in wajah[0].boxes:
        #     fx1, fy1, fx2, fy2 = map(
        #         int, face_box.xyxy[0].tolist()
        #     )

        #     face = body[fy1:fy2, fx1:fx2]

        #     if face.size == 0:
        #         continue

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

        # # Tampilkan kotak badan
        # cv2.rectangle(
        #     frame, (x1, y1), (x2, y2),
        #     (0, 255, 0), 2
        # )

        # # Tampilkan kotak wajah
        # for face_box in wajah[0].boxes:
        #     fx1, fy1, fx2, fy2 = map(
        #         int, face_box.xyxy[0].tolist()
        #     )

        #     cv2.rectangle(
        #         frame,
        #         (x1 + fx1, y1 + fy1),
        #         (x1 + fx2, y1 + fy2),
        #         (255, 0, 0), 2
        #     )