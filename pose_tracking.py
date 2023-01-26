import cv2
from media_pipe_module import mediapipe_drawing
from media_pipe_module import mediapipe_drawing_styles
from media_pipe_module import mediapipe_pose
import numpy as np
import math



def tracking(cap) :

    top = 0     # 코(results.pose_landmarks.landmark[0].y)
    toe = 0     # 왼쪽 발 끝(results.pose_landmarks.landmark[31].y)
    image_height = 0
    image_weight = 0

    array = [[0]*2 for i in range(33)]      #각 랜드마크별 xy좌표 저장 공간


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

            # mediapipe_drawing.plot_landmarks(
            #     results.pose_world_landmarks,
            #     mediapipe_pose.POSE_CONNECTIONS
            # )
            
            #영상 crop을 위해 높이에 대한 대략적인 정보를 보관합니다.
            if (results.pose_landmarks.landmark[0].y > top) :
                top = round(results.pose_landmarks.landmark[0].y * image_height)
            if (results.pose_landmarks.landmark[31].y > toe) :
                toe = round(results.pose_landmarks.landmark[31].y * image_height)


            #장준영의 코드

            #기준은 왼쪽 어깨(12)
            
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                

                array[id][0] = cx
                array[id][1] = cy
                
            # print("11", results.pose_landmarks.landmark[11])
            # print("12", results.pose_landmarks.landmark[12])

                #print(id, ":", cx, cy)
                

            _11_to_12 = np.array([[array[12][0]],[array[12][1]]]) - np.array([[array[11][0]],[array[11][1]]])    #11 -> 12    
            _14_to_12 = np.array([[array[12][0]],[array[12][1]]]) - np.array([[array[14][0]],[array[14][1]]])    #14 -> 12
            _13_to_11 = np.array([[array[11][0]],[array[11][1]]]) - np.array([[array[13][0]],[array[13][1]]])    #13 -> 11
            _16_to_14 = np.array([[array[14][0]],[array[14][1]]]) - np.array([[array[16][0]],[array[16][1]]])    #16 -> 14
            _15_to_13 = np.array([[array[13][0]],[array[13][1]]]) - np.array([[array[15][0]],[array[15][1]]])    #15 -> 13
            _18_to_16 = np.array([[array[16][0]],[array[16][1]]]) - np.array([[array[18][0]],[array[18][1]]])    #18 -> 16
            _17_to_15 = np.array([[array[15][0]],[array[15][1]]]) - np.array([[array[17][0]],[array[17][1]]])    #17 -> 15
            _20_to_16 = np.array([[array[16][0]],[array[16][1]]]) - np.array([[array[20][0]],[array[20][1]]])    #20 -> 16
            _19_to_15 = np.array([[array[15][0]],[array[15][1]]]) - np.array([[array[19][0]],[array[19][1]]])    #19 -> 15
            _22_to_16 = np.array([[array[16][0]],[array[16][1]]]) - np.array([[array[22][0]],[array[22][1]]])    #22 -> 16
            _21_to_15 = np.array([[array[15][0]],[array[15][1]]]) - np.array([[array[21][0]],[array[21][1]]])    #21 -> 15
            _24_to_12 = np.array([[array[12][0]],[array[12][1]]]) - np.array([[array[24][0]],[array[24][1]]])    #24 -> 12
            _23_to_11 = np.array([[array[11][0]],[array[11][1]]]) - np.array([[array[23][0]],[array[23][1]]])    #23 -> 11
            _23_to_24 = np.array([[array[24][0]],[array[24][1]]]) - np.array([[array[23][0]],[array[23][1]]])    #23 -> 24
            _26_to_24 = np.array([[array[24][0]],[array[24][1]]]) - np.array([[array[26][0]],[array[26][1]]])    #26 -> 24
            _25_to_23 = np.array([[array[23][0]],[array[23][1]]]) - np.array([[array[25][0]],[array[25][1]]])    #25 -> 23
            _28_to_26 = np.array([[array[26][0]],[array[26][1]]]) - np.array([[array[28][0]],[array[28][1]]])    #28 -> 26
            _27_to_25 = np.array([[array[25][0]],[array[25][1]]]) - np.array([[array[27][0]],[array[27][1]]])    #27 -> 25
            _30_to_28 = np.array([[array[28][0]],[array[28][1]]]) - np.array([[array[30][0]],[array[30][1]]])    #30 -> 28
            _29_to_27 = np.array([[array[27][0]],[array[27][1]]]) - np.array([[array[29][0]],[array[29][1]]])    #29 -> 27
            _32_to_28 = np.array([[array[28][0]],[array[28][1]]]) - np.array([[array[32][0]],[array[32][1]]])    #32 -> 28
            _31_to_27 = np.array([[array[27][0]],[array[27][1]]]) - np.array([[array[31][0]],[array[31][1]]])    #31 -> 27


            # print("1",_11_to_12)
            # print(_14_to_12)
            # print(_13_to_11)
            # print(_16_to_14)
            # print(_15_to_13)
            # print(_18_to_16)
            # print(_17_to_15)
            # print(_20_to_16)
            # print(_19_to_15)
            # print(_22_to_16)
            # print(_21_to_15)
            # print(_24_to_12)
            # print(_23_to_11)
            # print(_23_to_24)
            # print(_26_to_24)
            # print(_25_to_23)
            # print(_28_to_26)
            # print(_27_to_25)
            # print(_30_to_28)
            # print(_29_to_27)
            # print(_32_to_28)
            # print(_31_to_27)    


            #L2 정규화 함수
            def l2normalize(x, y):
                result = math.sqrt(x**2 + y**2)
                _x = x / result
                _y = y / result
                return _x, _y



            #l2 정규화 후 초기화
            l2_11_to_12 = np.array([l2normalize(_11_to_12[0], _11_to_12[1])])    #11 -> 12    
            l2_14_to_12 = np.array([l2normalize(_14_to_12[0], _14_to_12[1])])    #14 -> 12
            l2_13_to_11 = np.array([l2normalize(_13_to_11[0], _13_to_11[1])])    #13 -> 11
            l2_16_to_14 = np.array([l2normalize(_16_to_14[0], _16_to_14[1])])    #16 -> 14
            l2_15_to_13 = np.array([l2normalize(_15_to_13[0], _15_to_13[1])])    #15 -> 13
            l2_18_to_16 = np.array([l2normalize(_18_to_16[0], _18_to_16[1])])    #18 -> 16
            l2_17_to_15 = np.array([l2normalize(_17_to_15[0], _17_to_15[1])])    #17 -> 15
            l2_20_to_16 = np.array([l2normalize(_20_to_16[0], _20_to_16[1])])    #20 -> 16
            l2_19_to_15 = np.array([l2normalize(_19_to_15[0], _19_to_15[1])])    #19 -> 15
            l2_22_to_16 = np.array([l2normalize(_22_to_16[0], _22_to_16[1])])    #22 -> 16
            l2_21_to_15 = np.array([l2normalize(_21_to_15[0], _21_to_15[1])])    #21 -> 15
            l2_24_to_12 = np.array([l2normalize(_24_to_12[0], _24_to_12[1])])    #24 -> 12
            l2_23_to_11 = np.array([l2normalize(_23_to_11[0], _23_to_11[1])])    #23 -> 11
            l2_23_to_24 = np.array([l2normalize(_23_to_24[0], _23_to_24[1])])    #23 -> 24
            l2_26_to_24 = np.array([l2normalize(_26_to_24[0], _26_to_24[1])])    #26 -> 24
            l2_25_to_23 = np.array([l2normalize(_25_to_23[0], _25_to_23[1])])    #25 -> 23
            l2_28_to_26 = np.array([l2normalize(_28_to_26[0], _28_to_26[1])])    #28 -> 26
            l2_27_to_25 = np.array([l2normalize(_27_to_25[0], _27_to_25[1])])    #27 -> 25
            l2_30_to_28 = np.array([l2normalize(_30_to_28[0], _30_to_28[1])])    #30 -> 28
            l2_29_to_27 = np.array([l2normalize(_29_to_27[0], _29_to_27[1])])    #29 -> 27
            l2_32_to_28 = np.array([l2normalize(_32_to_28[0], _32_to_28[1])])    #32 -> 28
            l2_31_to_27 = np.array([l2normalize(_31_to_27[0], _31_to_27[1])])    #31 -> 27


            print("11 -> 12", l2_11_to_12)
            print("14 -> 12", l2_14_to_12)
            print("13 -> 11", l2_13_to_11)
            print("16 -> 14", l2_16_to_14)
            print("15 -> 13", l2_15_to_13)
            print("18 -> 16", l2_18_to_16)
            print("17 -> 15", l2_17_to_15)
            print("20 -> 16", l2_20_to_16)
            print("19 -> 15", l2_19_to_15)
            print("22 -> 16", l2_22_to_16)
            print("21 -> 15", l2_21_to_15)
            print("24 -> 12", l2_24_to_12)
            print("23 -> 11", l2_23_to_11)
            print("23 -> 24", l2_23_to_24)
            print("26 -> 24", l2_26_to_24)
            print("25 -> 23", l2_25_to_23)
            print("28 -> 26", l2_28_to_26)
            print("27 -> 25", l2_27_to_25)
            print("30 -> 28", l2_30_to_28)
            print("29 -> 27", l2_29_to_27)
            print("32 -> 28", l2_32_to_28)
            print("31 -> 27", l2_31_to_27)






            

            # 보기 편하게 이미지를 좌우 반전합니다. -> 영상은 좌우 반전 금지
            # 실제 사용에서는 성능향상을 목적으로 미리보기를 차단합니다.
            cv2.imshow('Pose_Check', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    print("최고 : ", top, ", 최저 : ", toe)
    return top, toe, image_weight