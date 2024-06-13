import cv2
import numpy as np
from ultralytics import YOLO

# モデルのロード
model = YOLO('yolov8n-pose.pt')

# 画像のロード
image1 = cv2.imread('ex1.jpg')
image2 = cv2.imread('ex1.jpg')

# 画像からキーポイントを抽出
results1 = model(image1)
results2 = model(image2)
keypoints1 = results1[0].keypoints
keypoints2 = results2[0].keypoints

# 類似度の計算関数
def pose_similarity(kp1, kp2):
    distances = []
    for i in range(5, 17):  # すべてのキーポイントを使用
        x1, y1 = kp1.data[0, i][0], kp1.data[0, i][1]
        x2, y2 = kp2.data[0, i][0], kp2.data[0, i][1]
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        distances.append(distance)
    return np.mean(distances)  

similarity = pose_similarity(keypoints1, keypoints2)
print(similarity)
