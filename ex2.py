import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
model = YOLO("yolov8x-pose.pt")

images = ['ex1.jpg', 'ex2_307.jpg', 'ex2_336.jpg', 'ex2_2015.jpg', 'ex2_3077.jpeg', 'ex2_5175.jpg']

results_list = []

for image in images:
    results = model(image)
    results_list.append(results)

keypoints_diff = []

for i in range(len(images) - 1):
    ex1_keypoints = results_list[0][0].keypoints
    keypoints_next = results_list[i + 1][0].keypoints
    
    diff = keypoints_next.data - ex1_keypoints.data
    diff_sum = np.sum(np.abs(diff.numpy()))  
    keypoints_diff.append((diff_sum, images[i], images[i + 1]))

keypoints_diff_sorted = sorted(keypoints_diff, key=lambda x: x[0])

sorted_images = []
for _, img1, img2 in keypoints_diff_sorted:
    if img1 not in sorted_images:
        sorted_images.append(img1)
    if img2 not in sorted_images:
        sorted_images.append(img2)

# 結果を出力
print("ex1.jpgからの違いが少ないもの順にソート")
for img in sorted_images:
    print(img)
