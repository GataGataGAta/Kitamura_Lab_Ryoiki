from ultralytics import YOLO
import cv2
import numpy as np

# ポーズ接続定義
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
image_path = "ex1.jpg"

# 基準画像からキーポイントを抽出
def get_reference_keypoints(image_path, model):
    image = cv2.imread(image_path)
    results = model(image)
    keypoints = results[0].keypoints
    return keypoints

def pose_similarity(ref_keypoints, keypoints, frame):
    distances = []
    center = [0, 0]
    ex1_center = [0, 0]
    diff = [0, 0]
    ex1_center[0] = int((abs(ref_keypoints.data[0, 5][0] - ref_keypoints.data[0, 6][0])/2) + ref_keypoints.data[0, 6][0])
    ex1_center[1] = int((abs(ref_keypoints.data[0, 5][1] - ref_keypoints.data[0, 12][1])/2) + ref_keypoints.data[0, 6][1])
    center[0] = int((abs(keypoints.data[0, 5][0] - keypoints.data[0, 6][0])/2) + keypoints.data[0, 6][0])
    center[1] = int((abs(keypoints.data[0, 5][1] - keypoints.data[0, 12][1])/2) + keypoints.data[0, 6][1])
    diff[0] = ex1_center[0] - center[0]
    diff[1] = ex1_center[1] - center[1]
    for i in range(5, 17):  
        ref_x, ref_y = ref_keypoints.data[0, i][0] - diff[0], ref_keypoints.data[0, i][1] - diff[1]
        x, y = keypoints.data[0, i][0], keypoints.data[0, i][1]
        distance = np.sqrt((ref_x - x) ** 2 + (ref_y - y) ** 2)
        distances.append(distance)
        cv2.circle(frame, (int(ref_x), int(ref_y)), 5, (0, 255, 0), thickness=3)
        start_pt = (int(ref_keypoints.data[0, i][0]), int(ref_keypoints.data[0, i][1]))
        end_pt = (int(ref_keypoints.data[0, i][0]), int(ref_keypoints.data[0, i][1]))
        cv2.line(frame, start_pt, end_pt, (0, 255, 0), thickness=2)

    cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), thickness=3)
    return np.mean(distances)


def pose_estimation_movie(filename_in, filename_out, model, ref_keypoints):
    cap = cv2.VideoCapture(filename_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(filename_out, fourcc, fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            keypoints = results[0].keypoints

            similarity = pose_similarity(ref_keypoints, keypoints, frame)
            print(similarity)
            if int(similarity) < 5 :
                color = (0, 0, 255) 
            else :
                color = (255, 0, 0)   

            for each in range(5, 17):
                x, y = int(keypoints.data[0, each][0]), int(keypoints.data[0, each][1])
                cv2.circle(frame, (x, y), 5, color, thickness=3, lineType=cv2.LINE_AA)

            for line in pose:
                start_pt = (int(keypoints.data[0, line[0]][0]), int(keypoints.data[0, line[0]][1]))
                end_pt = (int(keypoints.data[0, line[1]][0]), int(keypoints.data[0, line[1]][1]))
                cv2.line(frame, start_pt, end_pt, color, thickness=2)
            
            cv2.imshow("Movie", frame)
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
    ref_keypoints = get_reference_keypoints('ex1.jpg', model)
    pose_estimation_movie(filename_in, filename_out, model, ref_keypoints)
