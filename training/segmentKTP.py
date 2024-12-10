import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ultralytics import YOLO
import cv2
from PIL import Image
import random

sns.set(rc={'axes.facecolor': '#ffe4de'}, style='darkgrid')

dataDir = '/content/drive/MyDrive/KTP-new'

trainImagePath = os.path.join(dataDir, 'train', 'images')

imagesFiles = [f for f in os.listdir(trainImagePath) if f.endswith('.jpg')]

randomImages = random.sample(imagesFiles, 15)

plt.figure(figsize=(10, 10))

for i, image_file in enumerate(randomImages):
  image_path = os.path.join(trainImagePath, image_file)
  image = Image.open(image_path)
  plt.subplot(3, 5, i + 1)
  plt.imshow(image)
  plt.axis('off')

plt.suptitle('Random Selection of Dataset Images', fontsize = 24)

plt.tight_layout()
plt.show

model = YOLO('yolov8n-seg.pt')

yamlFilePath = os.path.join(dataDir, 'data.yaml')

results = model.train(
    data = yamlFilePath,
    epochs = 30,
    imgsz = 640,
    batch = 32,
    optimizer = 'auto',
    lr0 = 0.0001,
    lrf = 0.01,
    dropout = 0.25,
    device = 0,
    seed = 42
)

bestModelpath = '/content/runs/segment/train/weights/best.pt'
bestModel = YOLO(bestModelpath)

# Print the YOLO model architecture
print(bestModel)

# Count total parameters and trainable parameters
total_params = sum(p.numel() for p in bestModel.parameters())
trainable_params = sum(p.numel() for p in bestModel.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

validImagePath = os.path.join(dataDir, 'valid', 'images')

imageFiles = [f for f in os.listdir(validImagePath) if f.endswith('.jpg')]

numImages = len(imageFiles)
selectedImage = [imageFiles[i] for i in range(0, numImages, numImages // 9)]

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle('Validation Set Inferences')

for i, ax in enumerate(axes.flatten()):
  imagePath = os.path.join(validImagePath, selectedImage[i])
  results = bestModel.predict(source = imagePath, imgsz=640)
  annotatedImage = results[0].plot()
  annotatedImageRGB = cv2.cvtColor(annotatedImage, cv2.COLOR_BGR2RGB)
  ax.imshow(annotatedImageRGB)
  ax.axis('off')

plt.tight_layout()
plt.show()