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

for i in range(1, len(images)):
    ex1_keypoints = results_list[0][0].keypoints
    keypoints_next = results_list[i][0].keypoints
    
    diff = keypoints_next.data - ex1_keypoints.data
    diff_sum = np.sum(np.abs(diff.numpy()))  
    keypoints_diff.append((diff_sum, images[i]))
        

keypoints_diff_sorted = sorted(keypoints_diff, key=lambda x: x[0])
# sorted関数を使っソート(新しい配列を作成)

print("ソート結果")
for img in keypoints_diff_sorted:
    print(img[1])

