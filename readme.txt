I. MODEL DETECTION
1. Infuse watermark ở vị trí ngẫu nhiên để tạo dataset

python watermark_dataset.py -w data/images/watermarks -i data/images/VOC2012/JPEGImages \
-oo data/images/original -oi data/images/train -ol data/labels/train

python create_dataset.py -w ../yolov5/data/images/watermarks -i ../yolov5/data/images/VOC2012/JPEGImages -oo 
data/train/masks -oi data/train/imgs -n 30000 -s 12349

2. Train model detection:

python train.py --batch 16 --epochs 5 --data watermark.yaml --weights yolov5m.pt


3. Chạy detection và crop vùng watermark trên ảnh gốc và ảnh đã dính watermark để tạo dataset

python detect.py --weights runs/train/exp16/weights/best.pt --source data/images/train --i-mode 1


II. MODEL REMOVAL
4. Train model removal:
python train.py

5. Chạy infer (quay lại folder của model detection)
python detect.py --weights runs/train/exp16/weights/best.pt --source data/images/rever_test --i-mode 0


NOTE: Lúc chạy có thể sẽ báo 1 vài error ở mấy cái path, cái này là do một vài para sửa trực tiếp trên code, chưa đưa vào para của command
=> Sửa lại nếu cần thiết