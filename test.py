from ultralytics import YOLO
import cv2

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

def pose_estimation_movie(filename_in, filename_out, model):
    """ 動画ファイルからリアルタイム物体検出する関数 """
    cap = cv2.VideoCapture(filename_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(filename_out, fourcc, fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # フレームごとにキーポイント検出
            results = model(frame)
            keypoints = results[0].keypoints

            # キーポイントを基にポーズを描画
            for each in range(5, 17):
                x, y = int(keypoints.data[0,each][0]), int(keypoints.data[0,each][1])
                cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

            for line in pose:
                start_pt = (int(keypoints.data[0,line[0]][0]), int(keypoints.data[0,line[0]][1]))
                end_pt = (int(keypoints.data[0,line[1]][0]), int(keypoints.data[0,line[1]][1]))
                cv2.line(frame, start_pt, end_pt, (0, 0, 255), thickness=2)
            # 物体検出結果画像を表示
            cv2.imshow("Movie", frame)

            # 保存
            video.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    filename_in = 'ex3a.mp4'
    filename_out = 'pose-movie-out.mp4'
    model = YOLO('yolov8n-pose.pt')
    pose_estimation_movie(filename_in, filename_out, model)
