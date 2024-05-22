import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
# 画像の読み込み
img = cv2.imread('ex1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resize = cv2.resize(img, (384, 640))
cv2.rectangle(img, (640, 249), (648, 242), (255, 0, 0), thickness=20)
print(img.shape) # (3264, 4928, 3)
# 画像表示
plt.imshow(img)
plt.show()