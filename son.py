import cv2
from ultralytics import YOLO
import numpy as np

# YOLOモデルの読み込み
model = YOLO("yolov8x-pose.pt")

# 画像を読み込む
image = "ex1.jpg" 
movie = "ex3a.mp4"

path = movie
cap = cv2.VideoCapture(path)

i=1
while True :
    print("Frame: "+ str(i))
    #フレーム情報取得
    ret, img = cap.read()
    
    #動画が終われば処理終了
    if ret == False:
        break
    
    #動画表示
    cv2.imshow('Video', img)
    i +=1

cap.release()
cv2.destroyAllWindows()