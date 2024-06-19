from ultralytics import YOLO
import cv2

# モデルのロード
model = YOLO("yolov8x.pt")
results = model("ex4.jpg")

# 画像の読み込み
img = cv2.imread("ex4.jpg")

# 検出されたボックスを取得
boxes = results[0].boxes

# 各ボックスに対して円を描く
for box in boxes:
    # box.dataから座標を取得
    x_left = int(box.data[0][0])
    y_left = int(box.data[0][1])
    x_right = int(box.data[0][2])
    y_right = int(box.data[0][3])
    # 線を書く
    cv2.line(img, (x_left, y_left), (x_right, y_left),  (0, 0, 255) , thickness=5)
    cv2.line(img, (x_right, y_left), (x_right, y_right), (0, 0, 255), thickness=5 )
    cv2.line(img, (x_left, y_left), (x_left, y_right),  (0, 0, 255) , thickness=5)
    cv2.line(img, (x_left, y_right),(x_right, y_right),  (0, 0, 255), thickness=5 )
    # 確認用
    # cv2.circle(img, (x_left, y_left), 10, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    # cv2.circle(img, (x_right, y_left), 10, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    # cv2.circle(img, (x_right, y_right), 10, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    # cv2.circle(img, (x_left,  y_right), 10, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

# 画像の保存
cv2.imwrite("output.jpg", img)
