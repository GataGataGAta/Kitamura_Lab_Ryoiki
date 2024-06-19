from ultralytics import YOLO
model = YOLO("yolov8x-pose.pt")
f = open('runs/pose/predict7/labels/ex1.txt', 'r')
data = f.read()
print(data)
f.close()
results = model("ex4.jpg", save=True,
save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data)