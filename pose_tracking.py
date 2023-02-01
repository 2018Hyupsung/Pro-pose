import numpy as np
from numpy.linalg import norm
import math
import cv2
from media_pipe_module import mediapipe_drawing
from media_pipe_module import mediapipe_drawing_styles
from media_pipe_module import mediapipe_pose




def tracking(cap) :
    array = [[0]*2 for i in range(33)]      #각 랜드마크별 xy좌표 저장 공간
    connects = []   #랜드마크 사이 연결선용

    _, image = cap.read()
    height, weight, _ = image.shape

    normal = 'color=(0,255,0), thickness = 5'
    

    with mediapipe_pose.Pose(
        min_detection_confidence = 0.5,     #사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
        min_tracking_confidence = 0.5,      #포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
        model_complexity = 1,               #포즈 랜드마크 모델의 복합성입니다.
        ) as pose :

        
        while cap.isOpened() :
            success, image = cap.read()

            if not success :
                break                  #카메라 대신 동영상을 불러올 경우, break를 사용합니다.

            

            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)




            # 모든 랜드마크를 벡터화합니다.
            for idx, land in enumerate(results.pose_landmarks.landmark):
                if idx in [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]:
                    continue
                
                land_x, land_y = int(land.x*weight), int(land.y*height)
                

                array[idx][0] = land_x       # 해당 랜드마크의 x좌표입니다.
                array[idx][1] = land_y       # 해당 랜드마크의 y좌표입니다.
                connects.append(np.array([land_x, land_y]))         # 연결을 위해 두 랜드마크를 지정합니다.

            Ls_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[11][0],array[11][1]])    #11 -> 12    
            Re_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[14][0],array[14][1]])    #14 -> 12
            Le_Ls = np.array([array[11][0],array[11][1]]) - np.array([array[13][0],array[13][1]])    #13 -> 11
            Rw_Re = np.array([array[14][0],array[14][1]]) - np.array([array[16][0],array[16][1]])    #16 -> 14
            Lw_Le = np.array([array[13][0],array[13][1]]) - np.array([array[15][0],array[15][1]])    #15 -> 13
            Rh_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[24][0],array[24][1]])    #24 -> 12
            Lh_Ls = np.array([array[11][0],array[11][1]]) - np.array([array[23][0],array[23][1]])    #23 -> 11
            Lh_Rh = np.array([array[24][0],array[24][1]]) - np.array([array[23][0],array[23][1]])    #23 -> 24
            Rk_Rh = np.array([array[24][0],array[24][1]]) - np.array([array[26][0],array[26][1]])    #26 -> 24
            Lk_Lh = np.array([array[23][0],array[23][1]]) - np.array([array[25][0],array[25][1]])    #25 -> 23
            Ra_Rk = np.array([array[26][0],array[26][1]]) - np.array([array[28][0],array[28][1]])    #28 -> 26
            La_Lk = np.array([array[25][0],array[25][1]]) - np.array([array[27][0],array[27][1]])    #27 -> 25







            # L2 정규화
            # 단순성을 위한 L2 정규화 : https://codingrabbit.tistory.com/21
            # 정규화 : https://light-tree.tistory.com/125
            # 각 영상에서 두 랜드마크 사이의 벡터를 단위벡터로 표현.
            def L2normalize(x, y):
                result = math.sqrt(x**2 + y**2)
                _x = x / result
                _y = y / result
                return _x, _y
            
            L2_Ls_Rs = np.array(L2normalize(Ls_Rs[0], Ls_Rs[1]))            #11 -> 12    
            L2_Re_Rs = np.array(L2normalize(Re_Rs[0], Re_Rs[1]))            #14 -> 12
            L2_Le_Ls = np.array(L2normalize(Le_Ls[0], Le_Ls[1]))            #13 -> 11
            L2_Rw_Re = np.array(L2normalize(Rw_Re[0], Rw_Re[1]))            #16 -> 14
            L2_Lw_Le = np.array(L2normalize(Lw_Le[0], Lw_Le[1]))            #15 -> 13
            L2_Rh_Rs = np.array(L2normalize(Rh_Rs[0], Rh_Rs[1]))            #24 -> 12
            L2_Lh_Ls = np.array(L2normalize(Lh_Ls[0], Lh_Ls[1]))            #23 -> 11
            L2_Lh_Rh = np.array(L2normalize(Lh_Rh[0], Lh_Rh[1]))            #23 -> 24
            L2_Rk_Rh = np.array(L2normalize(Rk_Rh[0], Rk_Rh[1]))            #26 -> 24
            L2_Lk_Lh = np.array(L2normalize(Lk_Lh[0], Lk_Lh[1]))            #25 -> 23
            L2_Ra_Rk = np.array(L2normalize(Ra_Rk[0], Ra_Rk[1]))            #28 -> 26
            L2_La_Lk = np.array(L2normalize(La_Lk[0], La_Lk[1]))            #27 -> 25


            def cos_sim(a, b):      #코사인 유사도
                return np.dot(a, b) / (norm(a) * norm(b))

            cs1 = cos_sim(L2_Le_Ls, L2_Re_Rs)
            #cs2 = cos_sim(l2_11_to_12, l2_13_to_11)



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

            '''
            mediapipe_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mediapipe_pose.POSE_CONNECTIONS,
                
                #drawing_styles.py에서
                #get_default_pose_landmarks_style()의 속성을 변경합니다.
                
                #landmark_drawing_spec = mediapipe_drawing.DrawingSpec(color = (0,0,0), thickness = 8),
                #connection_drawing_spec = mediapipe_drawing.DrawingSpec(color=(0,255,0), thickness = 5),                                                             
                )
            '''
            '''
            cv2.line(
                image,
                (connects[0][0], connects[0][1]),
                (connects[1][0], connects[1][1]),
                color = (255,0,0),
                thickness = 8
            )
            '''
            cv2.line(
                image,
                (array[11][0], array[12][1]),
                (array[12][0], array[12][1]),
                color = (255,0,0),
                thickness = 7
            )

            if(cs1 < 0.7) :     #코사인 유사도가 70% 이하일 때
                cv2.line(
                image,
                (array[12][0], array[12][1]),
                (array[14][0], array[14][1]),
                color = (0,0,255),
                thickness = 7
            )
            

            # 보기 편하게 이미지를 좌우 반전합니다. -> 영상은 좌우 반전 금지
            # 실제 사용에서는 성능향상을 목적으로 미리보기를 차단합니다.
            cv2.imshow('Pose_Check', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
