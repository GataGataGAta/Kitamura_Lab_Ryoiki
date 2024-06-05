import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
model = YOLO("yolov8x-pose.pt")
results = model("mikky.jpg")
keypoints = results[0].keypoints

# 画像の読み込み
img = cv2.imread('mikky.jpg')

class data:
    id = 0
    x = 0
    y = 0

    def set(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y



listdata = []
for each in range(5, 17):
    A=data()
    A.set(each, int(keypoints.data[0,each][0]), int(keypoints.data[0,each][1]))
    listdata.append(A)


for ldata in listdata :
    cv2.circle(img, (ldata.x, ldata.y), 5, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

pose = [
    [10, 8],
    [8, 6],
    [9, 7],
    [7, 5],
    [16, 14],
    [14, 12],
    [15, 13],
    [13, 11],
    [12, 11],
    [6, 12],
    [5, 6],
    [11, 5]
]

for line in pose:
    cv2.line(img, (listdata[line[0] - 5].x, listdata[line[0] - 5].y), ( listdata[line[1] - 5].x, listdata[line[1] - 5].y), (0, 0, 0), thickness=5)
# 画像の確認
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()