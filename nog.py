from ultralytics import YOLO
import cv2
movie_path_in = "ex3a.mp4"
movie_path_out = "ex3.mp4"

cap_movie = cv2.VideoCapture(movie_path_in) #動画を読み込む
# print(cap_movie.isOpened())

frame_sum = cap_movie.get(cv2.CAP_PROP_FRAME_COUNT) #総フレーム数を取得
# print(frame_sum)
frame_now = cap_movie.get(cv2.CAP_PROP_POS_FRAMES) #現在のフレーム番号を取得
# print(frame_now)

count = 0 #フレームのカウンタ

frame_temp = []


while True: #リストにフレームごとの画像を保存
    ret, frame = cap_movie.read() #1進めたフレームの情報を取得
    if ret == True:
        frame_temp.append(frame)
        
    else:
        break
model = YOLO("yolov8x-pose.pt")



pose = [[10,8],[8,6],[6,12],[12,14],[14,16],[6,5],[5,7],[7,9],[5,11],[12,11],[11,13],[13,15]]

frame_after = []

for frame_num in frame_temp:
    
    results = model.predict(source=frame_temp)
    keypoints = results[0].keypoints


    for i in range(5,16):
        cv2.circle(frame_num, (int(keypoints.data[0][i][0]),int(keypoints.data[0][i][1])), 2,(255,244,34), thickness=2)

    for j in range(len(pose)):
        cv2.line(frame_num, (int(keypoints.data[0][pose[j][0]][0]), int(keypoints.data[0][pose[j][0]][1])),(int((keypoints.data[0][pose[j][1]][0])), int(keypoints.data[0][pose[j][1]][1])), (255, 0, 0))

    frame_after.append(frame_num)
    cv2.imshow("output",frame_after)