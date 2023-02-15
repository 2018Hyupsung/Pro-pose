import numpy as np
import pandas as pd
from numpy.linalg import norm
import math
import cv2
from media_pipe_module import mediapipe_drawing
from media_pipe_module import mediapipe_drawing_styles
from media_pipe_module import mediapipe_pose
from dtaidistance import dtw




# ins- 교수자(instructor)
# stu- 학습자(student)




# csv 불러오기
def read_ins_info(csv_path, instructor, info) :
    data_frame_raw = pd.read_csv(csv_path+instructor+info, index_col=0, na_values=['None'])
    data_frame_nan = data_frame_raw.replace({np.nan: None})
    data_frame = np.array(data_frame_nan)
    #data_frame = data_frame.astype('float64')
    return data_frame





def tracking(ins_info, stu_info, cap) :

    
    dtw_array_count = 0
    array = [[0]*2 for j in range(33)]      # (학생)각 랜드마크별 xy좌표 저장 공간
    scores = np.zeros(12)
    #(공통) 랜드마크 간 연결 리스트
    connects_list = [[11, 12], [14, 12], [13, 11], [16, 14], [15, 13], 
                     [24, 12], [23, 11], [23, 24], [26, 24], [25, 23], [28, 26], [27, 25]]



    # L2 정규화
    # 단순성을 위한 L2 정규화 : https://codingrabbit.tistory.com/21
    # 정규화 : https://light-tree.tistory.com/125
    # 각 영상에서 두 랜드마크 사이의 벡터를 단위벡터로 표현.
    def L2normalize(x, y):
        if ((x or y) == None) :
            return None, None
        result = math.sqrt(x**2 + y**2)
        _x = x / result
        _y = y / result
        return _x, _y
    


    # 코사인유사도 (-1 ~ 1)
    def cos_sim(a, b):
        if((a[0] is None) or (b[0] is None)) :
            return -2
        return np.dot(a, b) / (norm(a) * norm(b))
    

    # 유클리드 거리 (0 ~ 2)
    def euclid(cos) :
        if (cos == -2) :
            return np.nan
        if (2.0 * (1.0 - cos) < 0) :
            return 0
        return math.sqrt(2.0 * (1.0 - cos))


    # score
    def score(euc) :
        if (euc == np.nan) :
            return np.nan
        return 100 - (100 * (0.5 * euc))



    _, image = cap.read()
    height, weight, _ = image.shape

    

    with mediapipe_pose.Pose(
        min_detection_confidence = 0.5,     #사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
        min_tracking_confidence = 0.5,      #포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
        model_complexity = 1,               #포즈 랜드마크 모델의 복합성입니다.
        ) as pose :

            
        # 포즈 주석을 이미지 위에 그립니다.
        """
        drawing_utils.py
        line 157다음
        if idx in [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]:
            continue
        추가로 특정 랜드마크의 생성을 무시합니다.
        """
        #ins_image.flags.writeable = True
        #ins_image = cv2.cvtColor(ins_image, cv2.COLOR_RGB2BGR)

        '''
        mediapipe_drawing.draw_landmarks(
            ins_image,
            ins_results.pose_landmarks,
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
        '''
        
        '''
        # 보기 편하게 이미지를 좌우 반전합니다. -> 영상은 좌우 반전 금지
        # 실제 사용에서는 성능향상을 목적으로 미리보기를 차단합니다.
        #cv2.imshow('Pose_Check', ins_image)
        #if cv2.waitKey(5) & 0xFF == ord('q'):
        #    break
        
        




        # 이하 학생

        while cap.isOpened() :
            success, image = cap.read()

            if not success :
                break
            

            image.flags.writeable = True
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            

            try:
                results.pose_landmarks.landmark
            except AttributeError:
                continue

            #  # 모든 랜드마크를 벡터화합니다.
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
                
            
            # if((array[12][0] and array[12][1] and array[11][0] and array[11][1]) is not None) :
            #     Ls_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[11][0],array[11][1]])    #11 -> 12
            # else :
            #     Ls_Rs = np.array([None] * 2)    
            # if((array[12][0] and array[12][1] and array[14][0] and array[14][1]) is not None) :
            #     Re_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[14][0],array[14][1]])    #14 -> 12
            # else :
            #     Re_Rs = np.array([None] * 2)   
            # if((array[11][0] and array[11][1] and array[13][0] and array[13][1]) is not None) :
            #     Le_Ls = np.array([array[11][0],array[11][1]]) - np.array([array[13][0],array[13][1]])    #13 -> 11
            # else :
            #     Le_Ls = np.array([None] * 2)   
            # if((array[14][0] and array[14][1] and array[16][0] and array[16][1]) is not None) :
            #     Rw_Re = np.array([array[14][0],array[14][1]]) - np.array([array[16][0],array[16][1]])    #16 -> 14
            # else :
            #     Rw_Re = np.array([None] * 2)   
            # if((array[13][0] and array[13][1] and array[15][0] and array[15][1]) is not None) :
            #     Lw_Le = np.array([array[13][0],array[13][1]]) - np.array([array[15][0],array[15][1]])    #15 -> 13
            # else :
            #     Lw_Le = np.array([None] * 2)   
            # if((array[12][0] and array[12][1] and array[24][0] and array[24][1]) is not None) :
            #     Rh_Rs = np.array([array[12][0],array[12][1]]) - np.array([array[24][0],array[24][1]])    #24 -> 12
            # else :
            #     Rh_Rs = np.array([None] * 2)   
            # if((array[11][0] and array[11][1] and array[23][0] and array[23][1]) is not None) :
            #     Lh_Ls = np.array([array[11][0],array[11][1]]) - np.array([array[23][0],array[23][1]])    #23 -> 11
            # else :
            #     Lh_Ls = np.array([None] * 2)   
            # if((array[24][0] and array[24][1] and array[23][0] and array[23][1]) is not None) :
            #     Lh_Rh = np.array([array[24][0],array[24][1]]) - np.array([array[23][0],array[23][1]])    #23 -> 24
            # else :
            #     Lh_Rh = np.array([None] * 2)   
            # if((array[24][0] and array[24][1] and array[26][0] and array[26][1]) is not None) :
            #     Rk_Rh = np.array([array[24][0],array[24][1]]) - np.array([array[26][0],array[26][1]])    #26 -> 24
            # else :
            #     Rk_Rh = np.array([None] * 2)   
            # if((array[23][0] and array[23][1] and array[25][0] and array[25][1]) is not None) :
            #     Lk_Lh = np.array([array[23][0],array[23][1]]) - np.array([array[25][0],array[25][1]])    #25 -> 23
            # else :
            #     Lk_Lh = np.array([None] * 2)   
            # if((array[26][0] and array[26][1] and array[28][0] and array[28][1]) is not None) :
            #     Ra_Rk = np.array([array[26][0],array[26][1]]) - np.array([array[28][0],array[28][1]])    #28 -> 26
            # else :
            #     Ra_Rk = np.array([None] * 2)   
            # if((array[25][0] and array[25][1] and array[27][0] and array[27][1]) is not None) :
            #     La_Lk = np.array([array[25][0],array[25][1]]) - np.array([array[27][0],array[27][1]])    #27 -> 25
            # else :
            #     La_Lk = np.array([None] * 2)   



            # # L2 정규화
            # L2_Ls_Rs = np.array(L2normalize(Ls_Rs[0], Ls_Rs[1]))            #11 -> 12    
            # L2_Re_Rs = np.array(L2normalize(Re_Rs[0], Re_Rs[1]))            #14 -> 12
            # L2_Le_Ls = np.array(L2normalize(Le_Ls[0], Le_Ls[1]))            #13 -> 11
            # L2_Rw_Re = np.array(L2normalize(Rw_Re[0], Rw_Re[1]))            #16 -> 14
            # L2_Lw_Le = np.array(L2normalize(Lw_Le[0], Lw_Le[1]))            #15 -> 13
            # L2_Rh_Rs = np.array(L2normalize(Rh_Rs[0], Rh_Rs[1]))            #24 -> 12
            # L2_Lh_Ls = np.array(L2normalize(Lh_Ls[0], Lh_Ls[1]))            #23 -> 11
            # L2_Lh_Rh = np.array(L2normalize(Lh_Rh[0], Lh_Rh[1]))            #23 -> 24
            # L2_Rk_Rh = np.array(L2normalize(Rk_Rh[0], Rk_Rh[1]))            #26 -> 24
            # L2_Lk_Lh = np.array(L2normalize(Lk_Lh[0], Lk_Lh[1]))            #25 -> 23
            # L2_Ra_Rk = np.array(L2normalize(Ra_Rk[0], Ra_Rk[1]))            #28 -> 26
            # L2_La_Lk = np.array(L2normalize(La_Lk[0], La_Lk[1]))            #27 -> 25


            ins_dtw_info = [[] for i in range(12)]
            stu_dtw_info = [[] for i in range(12)]
            #각 랜드마크 벡터를 dtw를 7프레임마다 계산
            for i in range(dtw_array_count, dtw_array_count + 25):
                
                # if ins_info[dtw_array_count][0] == None:
                #     ins_info[dtw_array_count][0] = ins_info[0][0]
                    
                # if ins_info[dtw_array_count][1] == None:
                #     ins_info[dtw_array_count][1] = ins_info[0][1]

                ins_dtw_info[0].append(np.array([ins_info[i][0], ins_info[i][1]]))
                stu_dtw_info[0].append(np.array([stu_info[i][0], stu_info[i][1]]))
                ins_dtw_info[1].append(np.array([ins_info[i][2], ins_info[i][3]]))
                stu_dtw_info[1].append(np.array([stu_info[i][2], stu_info[i][3]]))
                ins_dtw_info[2].append(np.array([ins_info[i][4], ins_info[i][5]]))
                stu_dtw_info[2].append(np.array([stu_info[i][4], stu_info[i][5]]))
                ins_dtw_info[3].append(np.array([ins_info[i][6], ins_info[i][7]]))
                stu_dtw_info[3].append(np.array([stu_info[i][6], stu_info[i][7]]))
                ins_dtw_info[4].append(np.array([ins_info[i][8], ins_info[i][9]]))
                stu_dtw_info[4].append(np.array([stu_info[i][8], stu_info[i][9]]))
                ins_dtw_info[5].append(np.array([ins_info[i][10], ins_info[i][11]]))
                stu_dtw_info[5].append(np.array([stu_info[i][10], stu_info[i][11]]))
                ins_dtw_info[6].append(np.array([ins_info[i][12], ins_info[i][13]]))
                stu_dtw_info[6].append(np.array([stu_info[i][12], stu_info[i][13]]))
                ins_dtw_info[7].append(np.array([ins_info[i][14], ins_info[i][15]]))
                stu_dtw_info[7].append(np.array([stu_info[i][14], stu_info[i][15]]))
                ins_dtw_info[8].append(np.array([ins_info[i][16], ins_info[i][17]]))
                stu_dtw_info[8].append(np.array([stu_info[i][16], stu_info[i][17]]))
                ins_dtw_info[9].append(np.array([ins_info[i][18], ins_info[i][19]]))
                stu_dtw_info[9].append(np.array([stu_info[i][18], stu_info[i][19]]))
                ins_dtw_info[10].append(np.array([ins_info[i][20], ins_info[i][21]]))
                stu_dtw_info[10].append(np.array([stu_info[i][20], stu_info[i][21]]))
                ins_dtw_info[11].append(np.array([ins_info[i][22], ins_info[i][23]]))
                stu_dtw_info[11].append(np.array([stu_info[i][22], stu_info[i][23]]))
            
            dtw_array_count += 1

            for i in range(12):
                scores[i] = dtw.distance(ins_dtw_info[i], stu_dtw_info[i], window=3)
            print("score0", scores[0])
            print("score1", scores[1])
            print("score2", scores[2])
            print("score3", scores[3])
            print("score4", scores[4])
            print("score0", scores[5])
            print("score1", scores[6])
            print("score2", scores[7])
            print("score3", scores[8])
            print("score4", scores[9])
            print("score0", scores[10])
            print("score1", scores[11])
            
            # 코사인 유사도 및 유클리드 거리
            # cs1 = euclid(cos_sim(np.array([ins_info[ins_info_idx][0], ins_info[ins_info_idx][1]]), np.array([stu_info[stu_info_idx][0], stu_info[stu_info_idx][1]])))
            # scores[0] = dtw.distance()
            # cs2 = euclid(cos_sim(np.array([ins_info[ins_info_idx][2], ins_info[ins_info_idx][3]]), L2_Re_Rs))
            # scores[1] = score(cs2)
            # cs3 = euclid(cos_sim(np.array([ins_info[ins_info_idx][4], ins_info[ins_info_idx][5]]), L2_Le_Ls))
            # scores[2] = score(cs3)
            # cs4 = euclid(cos_sim(np.array([ins_info[ins_info_idx][6], ins_info[ins_info_idx][7]]), L2_Rw_Re))
            # scores[3] = score(cs4)
            # cs5 = euclid(cos_sim(np.array([ins_info[ins_info_idx][8], ins_info[ins_info_idx][9]]), L2_Lw_Le))
            # scores[4] = score(cs5)
            # cs6 = euclid(cos_sim(np.array([ins_info[ins_info_idx][10], ins_info[ins_info_idx][11]]), L2_Rh_Rs))
            # scores[5] = score(cs6)
            # cs7 = euclid(cos_sim(np.array([ins_info[ins_info_idx][12], ins_info[ins_info_idx][13]]), L2_Lh_Ls))
            # scores[6] = score(cs7)
            # cs8 = euclid(cos_sim(np.array([ins_info[ins_info_idx][14], ins_info[ins_info_idx][15]]), L2_Lh_Rh))
            # scores[7] = score(cs8)
            # cs9 = euclid(cos_sim(np.array([ins_info[ins_info_idx][16], ins_info[ins_info_idx][17]]), L2_Rk_Rh))
            # scores[8] = score(cs9)
            # cs10 = euclid(cos_sim(np.array([ins_info[ins_info_idx][18], ins_info[ins_info_idx][19]]), L2_Lk_Lh))
            # scores[9] = score(cs10)
            # cs11 = euclid(cos_sim(np.array([ins_info[ins_info_idx][20], ins_info[ins_info_idx][21]]), L2_Ra_Rk))
            # scores[10] = score(cs11)
            # cs12 = euclid(cos_sim(np.array([ins_info[ins_info_idx][22], ins_info[ins_info_idx][23]]), L2_La_Lk))
            # scores[11] = score(cs12)
            
            

            
            # print('Ls_Rs : ',scores[0],'%')
            # print('Re_Rs : ',scores[1],'%')
            # print('Le_Ls : ',scores[2],'%')
            # print('Rw_Re : ',scores[3],'%')
            # print('Lw_Le : ',scores[4],'%')
            # print('Rh_Rs : ',scores[5],'%')
            # print('Lh_Ls : ',scores[6],'%')
            # print('Lh_Rh : ',scores[7],'%')
            # print('Rk_Rh : ',scores[8],'%')
            # print('Lk_Lh : ',scores[9],'%')
            # print('Ra_Rk : ',scores[10],'%')
            # print('La_Lk : ',scores[11],'%')
            # print('Overall : ', np.nanmean(scores),'%')
            

            #cv2 - 랜드마크 선 표현
            for s_idx, i in enumerate(connects_list) :
                if array[i[0]][0] is not None and array[i[0]][1] is not None and array[i[1]][0] is not None and array[i[1]][1] is not None:
                    if scores[s_idx] < 5 :
                        color = (255, 0, 0)
                    elif scores[s_idx] < 15 :
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.line(
                    image,
                    (array[i[0]][0], array[i[0]][1]),
                    (array[i[1]][0], array[i[1]][1]),
                    color = color,
                    thickness = 7
                    )


            cv2.imshow('Pro-pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

