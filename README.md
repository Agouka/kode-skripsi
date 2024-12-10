# kode-skripsi
All the codes needed to complete my undergraduate thesis.

## Daftar Isi

1. [Dataset](#dataset)
2. [Model](#model)
3. [Hasil](#hasil)

## Dataset

Dataset akan dijelaskan jumlah dataset, pembagian train - valid - test, augmentasi dan preprocessing.

1. KTP
   - Sumber: Roboflow
   - Total Gambar: 1350
   - Pembagian: 70% train, 20% valid, 10% test
   - Preprocessing: Resize 640x640
   - Augmentasi: Rotasi 90°
                 Rotasi acak -15° hingga +15°
                 Penyesuaian Kecerahan -25% hingga 0%

## Model

Model akan menjelaskan hyperparameter dan lainnya berhubungan dengan model.
### Image Segmentation
**Tujuan**: Untuk prediksi lokasi KTP.

- Arsitektur: YOLOv8
- Dataset: 1350 gambar
- Performance:
  - Box mAP50: 0.995
  - Mask mAP50: 0.995
- Format File: PyTorch (.pt) dan NCNN (.ncnn)
