import numpy as np
import pandas as pd
import math
from media_pipe_module import mediapipe_pose

'''

이하 랜드마크 값 순서

Ls_Rs            #11 -> 12    
Re_Rs            #14 -> 12
Le_Ls            #13 -> 11
Rw_Re            #16 -> 14
Lw_Le            #15 -> 13
Rh_Rs            #24 -> 12
Lh_Ls            #23 -> 11
Lh_Rh            #23 -> 24
Rk_Rh            #26 -> 24
Lk_Lh            #25 -> 23
Ra_Rk            #28 -> 26
La_Lk            #27 -> 25

'''



def tracking_info(cap, frames, instructor) :

    # L2 정규화
    # 단순성을 위한 L2 정규화 : https://codingrabbit.tistory.com/21
    # 정규화 : https://light-tree.tistory.com/125
    # 각 영상에서 두 랜드마크 사이의 벡터를 단위벡터로 표현.
    def L2normalize(x, y):
        if (x and y) == 0:
            return 0, 0
        if ((x or y) == None) :
            return None, None
        result = math.sqrt(x**2 + y**2)
        _x = x / result
        _y = y / result
        return _x, _y
            



    array = [[0]*2 for i in range(33)]      #각 랜드마크별 xy좌표 저장 공간

    # (instructor.csv) - 저장을 위해 데이터프레임을 생성합니다. / 프레임별 데이터를 담기 위해 열을 생성합니다.
    cols = ['L2_Ls_Rs_0', 'L2_Ls_Rs_1', 'L2_Re_Rs_0', 'L2_Re_Rs_1', 'L2_Le_Ls_0', 'L2_Le_Ls_1', 'L2_Rw_Re_0', 'L2_Rw_Re_1', 'L2_Lw_Le_0', 
            'L2_Lw_Le_1', 'L2_Rh_Rs_0', 'L2_Rh_Rs_1', 'L2_Lh_Ls_0', 'L2_Lh_Ls_1', 'L2_Lh_Rh_0', 'L2_Lh_Rh_1', 'L2_Rk_Rh_0', 'L2_Rk_Rh_1', 
            'L2_Lk_Lh_0', 'L2_Lk_Lh_1', 'L2_Ra_Rk_0', 'L2_Ra_Rk_1', 'L2_La_Lk_0','L2_La_Lk_1']
    
    L2_landmarks = np.zeros([frames,24])
    l2_idx = 0
    
    

    _, image = cap.read()
    height, weight, _ = image.shape

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
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            try:
                results.pose_landmarks.landmark
            except AttributeError:
                L2_landmarks[l2_idx][0], L2_landmarks[l2_idx][1] = None, None             #11 -> 12    
                L2_landmarks[l2_idx][2], L2_landmarks[l2_idx][3] = None, None            #14 -> 12
                L2_landmarks[l2_idx][4], L2_landmarks[l2_idx][5] = None, None            #13 -> 11
                L2_landmarks[l2_idx][6], L2_landmarks[l2_idx][7] = None, None            #16 -> 14
                L2_landmarks[l2_idx][8], L2_landmarks[l2_idx][9] = None, None            #15 -> 13
                L2_landmarks[l2_idx][10], L2_landmarks[l2_idx][11] = None, None          #24 -> 12
                L2_landmarks[l2_idx][12], L2_landmarks[l2_idx][13] = None, None          #23 -> 11
                L2_landmarks[l2_idx][14], L2_landmarks[l2_idx][15] = None, None          #23 -> 24
                L2_landmarks[l2_idx][16], L2_landmarks[l2_idx][17] = None, None          #26 -> 24
                L2_landmarks[l2_idx][18], L2_landmarks[l2_idx][19] = None, None          #25 -> 23
                L2_landmarks[l2_idx][20], L2_landmarks[l2_idx][21] = None, None          #28 -> 26
                L2_landmarks[l2_idx][22], L2_landmarks[l2_idx][23] = None, None          #27 -> 25
                
                l2_idx += 1

                continue
            # 모든 랜드마크를 벡터화합니다.
            for idx, land in enumerate(results.pose_landmarks.landmark):
                if idx in [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]:
                    continue
                
                if (land.visibility < 0.8) :        # 랜드마크의 가시성 신뢰도가 80% 이하로 떨어지면 값을 None으로 변경합니다.
                    land_x = None
                    land_y = None
                else : 
                    land_x, land_y = int(land.x*weight), int(land.y*height)

                array[idx][0] = land_x       # 해당 랜드마크의 x좌표입니다.
                array[idx][1] = land_y       # 해당 랜드마크의 y좌표입니다.
                
            
            if((array[12][0] and array[12][1] and array[11][0] and array[11][1]) is not None) :
                Ls_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[11][0],array[11][1]])    #11 -> 12
            else :
                Ls_Rs = np.array([None] * 2)    
            if((array[12][0] and array[12][1] and array[14][0] and array[14][1]) is not None) :
                Re_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[14][0],array[14][1]])    #14 -> 12
            else :
                Re_Rs = np.array([None] * 2)   
            if((array[11][0] and array[11][1] and array[13][0] and array[13][1]) is not None) :
                Le_Ls = np.array([array[11][0],array[11][1]]) - np.array([array[13][0],array[13][1]])    #13 -> 11
            else :
                Le_Ls = np.array([None] * 2)   
            if((array[14][0] and array[14][1] and array[16][0] and array[16][1]) is not None) :
                Rw_Re = np.array([array[14][0],array[14][1]]) - np.array([array[16][0],array[16][1]])    #16 -> 14
            else :
                Rw_Re = np.array([None] * 2)   
            if((array[13][0] and array[13][1] and array[15][0] and array[15][1]) is not None) :
                Lw_Le = np.array([array[13][0],array[13][1]]) - np.array([array[15][0],array[15][1]])    #15 -> 13
            else :
                Lw_Le = np.array([None] * 2)   
            if((array[12][0] and array[12][1] and array[24][0] and array[24][1]) is not None) :
                Rh_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[24][0],array[24][1]])    #24 -> 12
            else :
                Rh_Rs = np.array([None] * 2)   
            if((array[11][0] and array[11][1] and array[23][0] and array[23][1]) is not None) :
                Lh_Ls = np.array([array[11][0],array[11][1]]) - np.array([array[23][0],array[23][1]])    #23 -> 11
            else :
                Lh_Ls = np.array([None] * 2)   
            if((array[24][0] and array[24][1] and array[23][0] and array[23][1]) is not None) :
                Lh_Rh = np.array([array[24][0],array[24][1]]) - np.array([array[23][0],array[23][1]])    #23 -> 24
            else :
                Lh_Rh = np.array([None] * 2)   
            if((array[24][0] and array[24][1] and array[26][0] and array[26][1]) is not None) :
                Rk_Rh = np.array([array[24][0],array[24][1]]) - np.array([array[26][0],array[26][1]])    #26 -> 24
            else :
                Rk_Rh = np.array([None] * 2)   
            if((array[23][0] and array[23][1] and array[25][0] and array[25][1]) is not None) :
                Lk_Lh = np.array([array[23][0],array[23][1]]) - np.array([array[25][0],array[25][1]])    #25 -> 23
            else :
                Lk_Lh = np.array([None] * 2)   
            if((array[26][0] and array[26][1] and array[28][0] and array[28][1]) is not None) :
                Ra_Rk = np.array([array[26][0],array[26][1]]) - np.array([array[28][0],array[28][1]])    #28 -> 26
            else :
                Ra_Rk = np.array([None] * 2)   
            if((array[25][0] and array[25][1] and array[27][0] and array[27][1]) is not None) :
                La_Lk = np.array([array[25][0],array[25][1]]) - np.array([array[27][0],array[27][1]])    #27 -> 25
            else :
                La_Lk = np.array([None] * 2)   
            

            
           
            # L2 정규화
            L2_landmarks[l2_idx][0], L2_landmarks[l2_idx][1] = L2normalize(Ls_Rs[0], Ls_Rs[1])             #11 -> 12    
            L2_landmarks[l2_idx][2], L2_landmarks[l2_idx][3] = L2normalize(Re_Rs[0], Re_Rs[1])            #14 -> 12
            L2_landmarks[l2_idx][4], L2_landmarks[l2_idx][5] = L2normalize(Le_Ls[0], Le_Ls[1])            #13 -> 11
            L2_landmarks[l2_idx][6], L2_landmarks[l2_idx][7] = L2normalize(Rw_Re[0], Rw_Re[1])            #16 -> 14
            L2_landmarks[l2_idx][8], L2_landmarks[l2_idx][9] = L2normalize(Lw_Le[0], Lw_Le[1])            #15 -> 13
            L2_landmarks[l2_idx][10], L2_landmarks[l2_idx][11] = L2normalize(Rh_Rs[0], Rh_Rs[1])          #24 -> 12
            L2_landmarks[l2_idx][12], L2_landmarks[l2_idx][13] = L2normalize(Lh_Ls[0], Lh_Ls[1])          #23 -> 11
            L2_landmarks[l2_idx][14], L2_landmarks[l2_idx][15] = L2normalize(Lh_Rh[0], Lh_Rh[1])          #23 -> 24
            L2_landmarks[l2_idx][16], L2_landmarks[l2_idx][17] = L2normalize(Rk_Rh[0], Rk_Rh[1])          #26 -> 24
            L2_landmarks[l2_idx][18], L2_landmarks[l2_idx][19] = L2normalize(Lk_Lh[0], Lk_Lh[1])          #25 -> 23
            L2_landmarks[l2_idx][20], L2_landmarks[l2_idx][21] = L2normalize(Ra_Rk[0], Ra_Rk[1])          #28 -> 26
            L2_landmarks[l2_idx][22], L2_landmarks[l2_idx][23] = L2normalize(La_Lk[0], La_Lk[1])          #27 -> 25
            
            l2_idx += 1
    

    data_frame = pd.DataFrame(L2_landmarks, columns = cols)
    data_frame = data_frame.astype(float).round(8)
    data_frame.to_csv('./csv/'+instructor+'_15fps_.csv', na_rep='None')


            