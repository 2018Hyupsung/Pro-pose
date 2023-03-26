import numpy as np
import pandas as pd
import math
import cv2
from media_pipe_module import mediapipe_pose


# 이하 랜드마크 값 순서

# Ls_Rs            11 -> 12
# Re_Rs            14 -> 12
# Le_Ls            13 -> 11
# Rw_Re            16 -> 14
# Lw_Le            15 -> 13
# Rh_Rs            24 -> 12
# Lh_Ls            23 -> 11
# Lh_Rh            23 -> 24
# Rk_Rh            26 -> 24
# Lk_Lh            25 -> 23
# Ra_Rk            28 -> 26
# La_Lk            27 -> 25





def tracking_info(path, start, end, is_stu) :


    # L2 정규화
    # 각 영상에서 두 랜드마크 사이의 벡터를 단위벡터로 표현.
    def L2normalize(x, y):
        if (x and y) == 0:
            return 0.0, 0.0
        if ((x or y) == None) :
            return None, None
        result = math.sqrt(x**2 + y**2)
        _x = x / result
        _y = y / result
        return _x, _y
            


    coord_pairs = [[12, 11], [12, 14], [11, 13], [14, 16], [13, 15], [12, 24], 
               [11, 23], [24, 23], [24, 26], [23, 25], [26, 28], [25, 27]]

    array = [[0]*2 for i in range(33)]      #각 랜드마크별 xy좌표 저장 공간
    if(is_stu == True) :
        array_set = [[0] * 24 for i in range(end-start+1)]
        array_idx = 0

    #프레임별 데이터를 담기 위해 열을 생성합니다.
    cols = ['L2_Ls_Rs_0', 'L2_Ls_Rs_1', 'L2_Re_Rs_0', 'L2_Re_Rs_1', 'L2_Le_Ls_0', 'L2_Le_Ls_1', 'L2_Rw_Re_0', 'L2_Rw_Re_1', 'L2_Lw_Le_0', 
            'L2_Lw_Le_1', 'L2_Rh_Rs_0', 'L2_Rh_Rs_1', 'L2_Lh_Ls_0', 'L2_Lh_Ls_1', 'L2_Lh_Rh_0', 'L2_Lh_Rh_1', 'L2_Rk_Rh_0', 'L2_Rk_Rh_1', 
            'L2_Lk_Lh_0', 'L2_Lk_Lh_1', 'L2_Ra_Rk_0', 'L2_Ra_Rk_1', 'L2_La_Lk_0','L2_La_Lk_1']

    cols_land = ['11_x','11_y','12_x','12_y','13_x','13_y','14_x','14_y','15_x','15_y','16_x','16_y','23_x','23_y','24_x','24_y',
                 '25_x','25_y','26_x','26_y','27_x','27_y','28_x','28_y']
    

    L2_landmarks = np.zeros([end-start+1,24])
    l2_idx = 0

    start_num = start
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)    #프레임 이동

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     #영상 높이/넓이 계산
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    with mediapipe_pose.Pose(
        min_detection_confidence = 0.5,     #사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
        min_tracking_confidence = 0.5,      #포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
        model_complexity = 1,               #포즈 랜드마크 모델의 복합성입니다.
        ) as pose :
       

       while ((cap.isOpened()) and (l2_idx < end-start+1)) :
            success, image = cap.read()

            if not success :
                break                  #카메라 대신 동영상을 불러올 경우, break를 사용합니다.

            

            # 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            image.flags.writeable = False
            #image.flags.writeable = True
            results = pose.process(image)

            try:
                results.pose_landmarks.landmark
            except AttributeError:
                continue

            
            # 모든 랜드마크를 벡터화합니다.
            for idx, land in enumerate(results.pose_landmarks.landmark):
                if idx in [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]:
                    continue

                if (land.visibility < 0.3) :        # 랜드마크의 가시성 신뢰도가 30% 이하로 떨어지면 값을 None으로 변경합니다.
                    land_x = None
                    land_y = None
                else : 
                    land_x, land_y = int(land.x*width), int(land.y*height)

                array[idx][0] = land_x       # 해당 랜드마크의 x좌표입니다.
                array[idx][1] = land_y       # 해당 랜드마크의 y좌표입니다.

                if (is_stu == True) :
                    array_set[l2_idx][array_idx*2] = land_x
                    array_set[l2_idx][array_idx*2+1] = land_y
                    array_idx += 1

                
            for idx1, pair in enumerate (coord_pairs):
                if all(array[pair[i]][j] is not None for i in range(2) for j in range(2)):
                    difference = np.array([array[pair[0]][0], array[pair[0]][1]]) - np.array([array[pair[1]][0], array[pair[1]][1]])
                else:
                    difference = np.array([None] * 2)
                L2_landmarks[l2_idx][idx1*2], L2_landmarks[l2_idx][idx1*2+1] = L2normalize(difference[0], difference[1]) 
            
            l2_idx += 1
            array_idx = 0

    cap.release()
    
    
    data_frame = pd.DataFrame(L2_landmarks, columns = cols)
    data_frame = data_frame.astype(float).round(8)
    formatted_num = "{:03d}".format(start_num)
    data_frame.to_csv('./temp_csv/'+(str)(formatted_num)+'_15fps_.csv', na_rep='None', index=False)

    if is_stu == True :
        data_frame1 = pd.DataFrame(array_set, columns=cols_land)
        data_frame1.to_csv('./temp_land_csv/'+(str)(formatted_num)+'_15fps_.csv', na_rep='None', index=False)
