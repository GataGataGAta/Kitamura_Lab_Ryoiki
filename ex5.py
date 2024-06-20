import cv2
from ultralytics import YOLO

# YOLOv8モデルをロード
model = YOLO("yolov8n.pt")

# ビデオファイルを開く
video_path = "ex5.mp4"
cap = cv2.VideoCapture(video_path)

# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    success, frame = cap.read()

    if success:
        # フレームでYOLOv8トラッキングを実行し、フレーム間でトラックを永続化
        results = model.track(frame, persist=True)
        
        # 各ボックスに対して処理
        for result in results:
            for box in result.boxes:
                # box.dataから座標を取得
                x_left = int(box.data[0][0])
                y_left = int(box.data[0][1])
                x_right = int(box.data[0][2])
                y_right = int(box.data[0][3])
                
                # 各頂点に円を描画
                cv2.line(frame, (x_left, y_left), (x_right, y_left),  (0, 0, 255) , thickness=5)
                cv2.line(frame, (x_right, y_left), (x_right, y_right), (0, 0, 255), thickness=5 )
                cv2.line(frame, (x_left, y_left), (x_left, y_right),  (0, 0, 255) , thickness=5)
                cv2.line(frame, (x_left, y_right),(x_right, y_right),  (0, 0, 255), thickness=5 )

        # 注釈付きのフレームを表示
        cv2.imshow("YOLOv8トラッキング", frame)

        # 'q'が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break

# ビデオキャプチャオブジェクトを解放し、表示ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
