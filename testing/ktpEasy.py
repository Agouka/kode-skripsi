import cv2
import csv
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
from ultralytics import YOLO
import easyocr
import time
import numpy as np
import psutil

# Load model YOLOv8
model = YOLO('models/best-new.pt')

# Inisialisasi EasyOCR
reader = easyocr.Reader(['id'])

# Variabel global untuk kontrol alur
stop_processing = False
capture_times = []
prev_time = time.time()

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

                if class_id == 0:
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID Card: {confidence:.2f}', (int(xmin), int(ymin) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Menghitung FPS
                    curr_time = time.time()
                    time_diff = curr_time - prev_time
                    fps = 1 / time_diff if time_diff > 0 else 0
                    prev_time = curr_time
                    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    if not stop_processing:
        root.after(10, capture_and_detect)

# Fungsi untuk preprocessing gambar
def preprocess_image(img):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# Fungsi untuk tombol capture
def capture_button_click():
    global stop_processing, capture_times

    ret, frame = cap.read()
    if ret:
        cpu_before = psutil.cpu_percent()
        ram_before = psutil.virtual_memory().percent

        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls.cpu().numpy().item())

                if class_id == 0:
                    id_card_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                    processed_img = preprocess_image(id_card_img)

                    captured_img = Image.fromarray(processed_img)
                    captured_imgtk = ImageTk.PhotoImage(image=captured_img)
                    captured_label.imgtk = captured_imgtk
                    captured_label.config(image=captured_imgtk)

                    start_time = time.time()
                    ocr_result = reader.readtext(processed_img)
                    if len(ocr_result) >= 6:
                        fourth_line = ocr_result[3][1]
                        sixth_line = ocr_result[5][1]

                        # Ubah file path sesuai pencahayaan
                        # Berakhiran d artinya cahaya redup
                        # Berakhiran dd artinya cahaya minim
                        # Tidak ada akhiran artinya cahaya terang 
                        with open('results/easyocr/captured_data-dd.csv', mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            writer.writerow([fourth_line, sixth_line])
                        print(f"NIK: {fourth_line}, Nama: {sixth_line}")

                    end_time = time.time()
                    capture_time = end_time - start_time
                    capture_times.append(capture_time)
                    print(f"OCR Time: {capture_time:.4f} seconds")
                    ocr_time_label.config(text=f"Last Capture OCR Time: {capture_time:.4f} seconds")

                    avg_time = np.mean(capture_times) if capture_times else 0
                    avg_time_label.config(text=f"Average OCR Time: {avg_time:.4f} seconds")

        cpu_after = psutil.cpu_percent()
        ram_after = psutil.virtual_memory().percent
        print(f"CPU Usage Before: {cpu_before}% | CPU Usage After: {cpu_after}%")
        print(f"RAM Usage Before: {ram_before}% | RAM Usage After: {ram_after}%")

        cpu_label.config(text=f"CPU Usage: {cpu_after}%")
        ram_label.config(text=f"RAM Usage: {ram_after}%")
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

ocr_time_label = Label(root, text="Last Capture OCR Time: Not yet computed", font=('Helvetica', 12))
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

capture_and_detect()

root.mainloop()
