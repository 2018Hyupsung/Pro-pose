import cv2
from media_pipe_module import mediapipe_drawing
from media_pipe_module import mediapipe_drawing_styles
from media_pipe_module import mediapipe_pose




def tracking(cap) :

    top = 0     # 코(results.pose_landmarks.landmark[0].y)
    toe = 0     # 왼쪽 발 끝(results.pose_landmarks.landmark[31].y)
    image_height = 0
    image_weight = 0


    with mediapipe_pose.Pose(
        min_detection_confidence = 0.5,     #사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
        min_tracking_confidence = 0.5,      #포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
        model_complexity = 1,               #포즈 랜드마크 모델의 복합성입니다.
        ) as pose :

        # 영상 crop을 위해 영상 해상도에 대한 값을 저장합니다.
        _, image = cap.read()
        image_height, image_weight, _ = image.shape

        while cap.isOpened() :
            success, image = cap.read()

            if not success :
                break                  #카메라 대신 동영상을 불러올 경우, break를 사용합니다.

            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 포즈 주석을 이미지 위에 그립니다.
            """
            drawing_utils.py
            line 157다음
            if idx in [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]:
                continue
            추가로 특정 랜드마크의 생성을 무시합니다.
            """
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            mediapipe_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mediapipe_pose.POSE_CONNECTIONS,
                
                #drawing_styles.py에서
                #get_default_pose_landmarks_style()의 속성을 변경합니다.
                
                landmark_drawing_spec = mediapipe_drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec = mediapipe_drawing.DrawingSpec(color=(0,255,0), thickness=5)
                )
            
            #영상 crop을 위해 높이에 대한 대략적인 정보를 보관합니다.
            if (results.pose_landmarks.landmark[0].y > top) :
                top = round(results.pose_landmarks.landmark[0].y * image_height)
            if (results.pose_landmarks.landmark[31].y > toe) :
                toe = round(results.pose_landmarks.landmark[31].y * image_height)
            

            # 보기 편하게 이미지를 좌우 반전합니다. -> 영상은 좌우 반전 금지
            # 실제 사용에서는 성능향상을 목적으로 미리보기를 차단합니다.
            cv2.imshow('Pose_Check', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    print("최고 : ", top, ", 최저 : ", toe)
    return top, toe, image_weight