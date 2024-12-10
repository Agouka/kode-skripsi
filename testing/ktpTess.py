import cv2
import csv
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
from ultralytics import YOLO
import pytesseract
import time
import numpy as np
import re
import psutil

# Path ke Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load model YOLOv8
model = YOLO('models/best-new.pt')

# Variabel global
stop_processing = False
capture_times = []
prev_time = time.time()

# Fungsi untuk preprocessing gambar
def preprocess_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

# Lakukan OCR dan bersihkan hasilnya
def perform_ocr_with_tesseract(processed_img):
    gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    ocr_result = pytesseract.image_to_string(gray_img, lang='ind', config=custom_config)

    nik_match = re.search(r'NIK[:\s]*([\d]+)', ocr_result, re.IGNORECASE)
    nama_match = re.search(r'(Nama|NAMA)[:\s]*(.+)', ocr_result, re.IGNORECASE)

    nik = clean_ocr_result(nik_match.group(1)) if nik_match else "NIK Not Found"
    nama = clean_ocr_result(nama_match.group(2)) if nama_match else "Nama Not Found"
    
    return nik, nama

# Bersihkan hasil OCR dari karakter yang tidak diinginkan
def clean_ocr_result(text):
    return re.sub(r'[^A-Za-z0-9\s]', '', text).strip()

# Deteksi ID Card, pra-pemrosesan, OCR, dan simpan ke CSV
def detect_id_card_and_save(frame):
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls.cpu().numpy().item())

            if class_id == 0:  # ID card class
                id_card_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                processed_img = preprocess_image(id_card_img)

                start_time = time.time()
                nik, nama = perform_ocr_with_tesseract(processed_img)
                end_time = time.time()

                capture_time = end_time - start_time
                capture_times.append(capture_time)

                # Ubah file path sesuai pencahayaan
                # Berakhiran d artinya cahaya redup
                # Berakhiran dd artinya cahaya minim
                # Tidak ada akhiran artinya cahaya terang 
                with open('results/pytesseract/pt-captured_data-dd.csv', mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([nik, nama])

                return processed_img, nik, nama, capture_time

# Fungsi untuk menangkap dan mendeteksi ID Card
def capture_and_detect():
    global stop_processing, prev_time

    ret, frame = cap.read()
    if ret:
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                confidence = box.conf.cpu().numpy().item()
                class_id = int(box.cls.cpu().numpy().item())

                if class_id == 0:  # ID card class
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID Card: {confidence:.2f}', (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Hitung FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Tampilkan frame di Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    if not stop_processing:
        root.after(10, capture_and_detect)

# Fungsi untuk tombol capture
def capture_button_click():
    global stop_processing

    ret, frame = cap.read()
    if ret:
        processed_img, nik, nama, capture_time = detect_id_card_and_save(frame)

        if processed_img is not None:
            captured_img = Image.fromarray(processed_img)
            captured_imgtk = ImageTk.PhotoImage(image=captured_img)
            captured_label.imgtk = captured_imgtk
            captured_label.config(image=captured_imgtk)

            ocr_time_label.config(text=f"Last OCR Time: {capture_time:.4f} seconds")

            avg_time = np.mean(capture_times) if capture_times else 0
            avg_time_label.config(text=f"Average OCR Time: {avg_time:.4f} seconds")

            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent

            cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
            ram_label.config(text=f"RAM Usage: {ram_usage}%")
    else:
        print("Gagal menangkap gambar.")

# GUI tkinter
root = tk.Tk()
root.title("Webcam Capture")

frame_container = tk.Frame(root)
frame_container.pack()

video_label = Label(frame_container)
video_label.pack(side=tk.LEFT)

captured_label = Label(frame_container)
captured_label.pack(side=tk.RIGHT)

capture_button = Button(root, text="Capture", command=capture_button_click)
capture_button.pack()

ocr_time_label = Label(root, text="Last OCR Time: Not yet computed", font=('Helvetica', 12))
ocr_time_label.pack()

avg_time_label = Label(root, text="Average OCR Time: Not yet computed", font=('Helvetica', 12))
avg_time_label.pack()

cpu_label = Label(root, text="CPU Usage: Not yet computed", font=('Helvetica', 12))
cpu_label.pack()

ram_label = Label(root, text="RAM Usage: Not yet computed", font=('Helvetica', 12))
ram_label.pack()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

root.after(10, capture_and_detect)
root.mainloop()

cap.release()
