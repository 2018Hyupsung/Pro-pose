import cv2
import mediapipe
import numpy
from mediapipe.framework.formats import landmark_pb2 as mediapipe_landmarks     #랜드마크 정보를 가져옵니다.


mediapipe_drawing = mediapipe.solutions.drawing_utils               #포즈 주석을 그립니다.
mediapipe_drawing_styles = mediapipe.solutions.drawing_styles       #포즈 주석을 그릴 스타일을 지정합니다.
mediapipe_pose = mediapipe.solutions.pose                           #미디어파이프의 포즈 구성을 활용합니다.
#bodyconnections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
#(23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]
bodyconnections = [(0,1), (0,2), (2,4), (1,3), (3,5), (0,6), (1,7), (6,7), (6,8), (7,9), (8,10), (9,11), (10,12),
(11,13), (12,14), (13,15), (10,14), (11,15)]



# 웹캠 초기화. 오류가 발생한다면 -1, 0, 1 중 하나로 초기화 시켜봅니다.
cap = cv2.VideoCapture(1)

with mediapipe_pose.Pose(
    min_detection_confidence = 0.5,     #사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
    min_tracking_confidence = 0.5,      #포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
    model_complexity = 1,               #포즈 랜드마크 모델의 복합성입니다.
    ) as pose :

    while cap.isOpened() :
        success, image = cap.read()
        if not success :
            print("카메라를 찾을 수 없습니다.")
            continue                    #카메라 대신 동영상을 불러올 경우, break를 사용합니다.

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 꼭 필요한 랜드마크만 불러올 수 있게 정보를 불러옵니다.
        landmark_subset = mediapipe_landmarks.NormalizedLandmarkList(
        landmark=[
        results.pose_landmarks.landmark[11],
        results.pose_landmarks.landmark[12],
        results.pose_landmarks.landmark[13],
        results.pose_landmarks.landmark[14],
        results.pose_landmarks.landmark[15],
        results.pose_landmarks.landmark[16],
        #results.pose_landmarks.landmark[17],
        #results.pose_landmarks.landmark[18],
        #results.pose_landmarks.landmark[19],
        #results.pose_landmarks.landmark[20],
        #results.pose_landmarks.landmark[21],
        #results.pose_landmarks.landmark[22],
        results.pose_landmarks.landmark[23],
        results.pose_landmarks.landmark[24],
        results.pose_landmarks.landmark[25],
        results.pose_landmarks.landmark[26],
        results.pose_landmarks.landmark[27],
        results.pose_landmarks.landmark[28],
        results.pose_landmarks.landmark[29],
        results.pose_landmarks.landmark[30],
        results.pose_landmarks.landmark[31],
        results.pose_landmarks.landmark[32] 
        ]
        )

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mediapipe_drawing.draw_landmarks(
            image = image,
            landmark_list = landmark_subset,
            #results.pose_landmarks,
            #connections = bodyconnections,
            #landmark_drawing_spec = mediapipe_drawing_styles.get_default_pose_landmarks_style(),
            #connection_drawing_spec = mediapipe_drawing_styles.hand_connection_drawing_spec
            )

        #가져온 랜드마크를 토대로 선을 그립니다.
        poses = landmark_subset.landmark
        for i in range(0, len(bodyconnections)):
            start_idx = [
                poses[bodyconnections[i][0]].x,
                poses[bodyconnections[i][0]].y
            ]

            end_idx = [
                poses[bodyconnections[i][1]].x,
                poses[bodyconnections[i][1]].y
            ]
            IMG_HEIGHT, IMG_WIDTH = image.shape[:2]

            cv2.line(image,
                tuple(numpy.multiply(start_idx[:2], [
                    IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                tuple(numpy.multiply(end_idx[:2], [
                    IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                (255, 0, 0), 9)
        
        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('Pose_Check', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()