import numpy as np
import pandas as pd
from numpy.linalg import norm
import math
import cv2
from media_pipe_module import mediapipe_pose
from dtaidistance import dtw

font_italic = "FONT_ITALIC"


# ins- 교수자(instructor)
# stu- 학습자(student)


# csv 불러오기
def read_ins_info(csv_path, instructor, info) :
    data_frame_raw = pd.read_csv(csv_path+instructor+info, index_col=0, na_values=['None'])
    data_frame_nan = data_frame_raw.replace({np.nan: None})
    data_frame = np.array(data_frame_nan)
    return data_frame


# 코사인유사도 (-1 ~ 1) -----> dtaidistance dtw.py 289~300 line
def cos_sim(a, b):      
    if(a[0] is None) :
        a[0] = 999
    if(b[0] is None):
         b[0] = 999
    if(a[1] is None):
        a[1] = 999
    if(b[1] is None):
        b[1] = 999
    if((a[0] == 0 and a[1] == 0) or (b[0] == 0 and b[1] == 0)):
        return 1
    return np.dot(a, b) / (norm(a) * norm(b))
    

# 유클리드 거리 (0 ~ 2) -----> dtaidistance dtw.py 302~306 line
def euclid(cos) :
    if (cos == -2) :
        return 1e9
    if (2.0 * (1.0 - cos) < 0) :
        return 0
    return math.sqrt(2.0 * (1.0 - cos))




def tracking(ins_info, stu_info, cap, frame_total) :

    frame_now = 1
    dtw_array_count = 0
    dtw_how = 0
    array = [[0]*2 for j in range(33)]      # (학생)각 랜드마크별 xy좌표 저장 공간
    
    #(공통) 랜드마크 간 연결 리스트
    connects_list = [[11, 12], [14, 12], [13, 11], [16, 14], [15, 13], 
                     [24, 12], [23, 11], [23, 24], [26, 24], [25, 23], [28, 26], [27, 25]]

    

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


        while (cap.isOpened()) :
            success, image = cap.read()

            if not success :
                break
            

            image.flags.writeable = True
            results = pose.process(image)
            

            try:
                results.pose_landmarks.landmark
            except AttributeError:
                continue

            #  # 모든 랜드마크를 벡터화합니다.
            for idx, land in enumerate(results.pose_landmarks.landmark):
                if idx in [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]:
                    continue
                
                if (land.visibility < 0.3) :        # 랜드마크의 가시성 신뢰도가 80% 이하로 떨어지면 값을 None으로 변경합니다.
                    land_x = None
                    land_y = None
                else : 
                    land_x, land_y = int(land.x*weight), int(land.y*height)

                array[idx][0] = land_x       # 해당 랜드마크의 x좌표입니다.
                array[idx][1] = land_y       # 해당 랜드마크의 y좌표입니다.

            #stu의 dtw를 10프레임을 고정으로 하고 비교한다.
            #ins의 dtw를 40프레임 범위에서 비교하도록 한다.
            #15fps  현재 40프레임을 10프레임씩 총 31번 비교
            dtw_how = 10 # stu
            dtw_range = 30 # ins
            
            
            if(frame_total - frame_now > dtw_how) :
                stu_dtw_info = [[] for i in range(12)]
            
            # frame_info <= 15 인 경우에는 현재 프레임 포함 45프레임을 검사
            # frame_info > 15 인 경우에는 현재 프레임 이전 15프레임 + 이후 30프레임을 10프레임씩 검사
            scores = np.zeros(12)
            temp = np.zeros(12)
            min_scores = 999    # 해당 프레임의 dtw 최솟값
            ins_min_frames = 0  #dtw가 최소가되는 프레임의 시작값
            stu_min_frames = 0
            
            min_part_dtw = np.zeros(12) #dtw 평균이 최소가 되는 프레임의 부위별 dtw값

            # min_dtw = np.zeros(12)
            
            
            if(frame_total - frame_now > dtw_how) :
                for i in range(dtw_array_count, dtw_array_count + dtw_how):
                    stu_dtw_info[0].append(np.array([stu_info[i][0], stu_info[i][1]]))
                    stu_dtw_info[1].append(np.array([stu_info[i][2], stu_info[i][3]]))
                    stu_dtw_info[2].append(np.array([stu_info[i][4], stu_info[i][5]]))
                    stu_dtw_info[3].append(np.array([stu_info[i][6], stu_info[i][7]]))
                    stu_dtw_info[4].append(np.array([stu_info[i][8], stu_info[i][9]]))
                    stu_dtw_info[5].append(np.array([stu_info[i][10], stu_info[i][11]]))
                    stu_dtw_info[6].append(np.array([stu_info[i][12], stu_info[i][13]]))
                    stu_dtw_info[7].append(np.array([stu_info[i][14], stu_info[i][15]]))
                    stu_dtw_info[8].append(np.array([stu_info[i][16], stu_info[i][17]]))
                    stu_dtw_info[9].append(np.array([stu_info[i][18], stu_info[i][19]]))
                    stu_dtw_info[10].append(np.array([stu_info[i][20], stu_info[i][21]]))
                    stu_dtw_info[11].append(np.array([stu_info[i][22], stu_info[i][23]]))
            
            if(frame_now <= 15):    #현재 프레임이 15 이하인 경우 1 ~ 45프레임까지 비교
                for j in range(dtw_range - dtw_how + 1):
                    ins_dtw_info = [[] for i in range(12)]
                    for i in range(dtw_array_count + j, (dtw_array_count + j) + dtw_how):
                        ins_dtw_info[0].append(np.array([ins_info[i][0], ins_info[i][1]]))
                        ins_dtw_info[1].append(np.array([ins_info[i][2], ins_info[i][3]]))
                        ins_dtw_info[2].append(np.array([ins_info[i][4], ins_info[i][5]]))
                        ins_dtw_info[3].append(np.array([ins_info[i][6], ins_info[i][7]]))
                        ins_dtw_info[4].append(np.array([ins_info[i][8], ins_info[i][9]]))
                        ins_dtw_info[5].append(np.array([ins_info[i][10], ins_info[i][11]]))
                        ins_dtw_info[6].append(np.array([ins_info[i][12], ins_info[i][13]]))
                        ins_dtw_info[7].append(np.array([ins_info[i][14], ins_info[i][15]]))
                        ins_dtw_info[8].append(np.array([ins_info[i][16], ins_info[i][17]]))
                        ins_dtw_info[9].append(np.array([ins_info[i][18], ins_info[i][19]]))
                        ins_dtw_info[10].append(np.array([ins_info[i][20], ins_info[i][21]]))
                        ins_dtw_info[11].append(np.array([ins_info[i][22], ins_info[i][23]]))
                    for i in range(12):
                        temp[i] = dtw.distance(ins_dtw_info[i], stu_dtw_info[i], window=3)
                    average = np.mean(temp)
                    if(min_scores > average):
                        min_scores = average
                        ins_min_frames = dtw_array_count + j
                        stu_min_frames = dtw_array_count
                        min_part_dtw = (1 - (temp / 20)) * 100
                        # min_dtw = temp

            elif(frame_now > 15):   #현재 프레임이 15 초과인 경우 (현재 프레임 -15) 부터 (현재 프레임 + 30까지 비교)
                for j in range(-15 , dtw_range - dtw_how + 1 - 15):
                    ins_dtw_info = [[] for i in range(12)]
                    for i in range(dtw_array_count + j, (dtw_array_count + j) + dtw_how):
                        ins_dtw_info[0].append(np.array([ins_info[i][0], ins_info[i][1]]))
                        ins_dtw_info[1].append(np.array([ins_info[i][2], ins_info[i][3]]))
                        ins_dtw_info[2].append(np.array([ins_info[i][4], ins_info[i][5]]))
                        ins_dtw_info[3].append(np.array([ins_info[i][6], ins_info[i][7]]))
                        ins_dtw_info[4].append(np.array([ins_info[i][8], ins_info[i][9]]))
                        ins_dtw_info[5].append(np.array([ins_info[i][10], ins_info[i][11]]))
                        ins_dtw_info[6].append(np.array([ins_info[i][12], ins_info[i][13]]))
                        ins_dtw_info[7].append(np.array([ins_info[i][14], ins_info[i][15]]))
                        ins_dtw_info[8].append(np.array([ins_info[i][16], ins_info[i][17]]))
                        ins_dtw_info[9].append(np.array([ins_info[i][18], ins_info[i][19]]))
                        ins_dtw_info[10].append(np.array([ins_info[i][20], ins_info[i][21]]))
                        ins_dtw_info[11].append(np.array([ins_info[i][22], ins_info[i][23]]))
                    for i in range(12):
                        temp[i] = dtw.distance(ins_dtw_info[i], stu_dtw_info[i], window=3)
                    average = np.mean(temp)
                    if(min_scores > average):
                        min_scores = average
                        ins_min_frames = dtw_array_count + j
                        stu_min_frames = dtw_array_count
                        min_part_dtw = (1 - (temp / 20)) * 100
                        # min_dtw = temp

            
            # print(min_scores)
            # print(ins_min_frames)
            # print(stu_min_frames)
            print(min_part_dtw)
            average_min_dtw = np.mean(min_part_dtw)
            print(average_min_dtw)
                
                
               
            dtw_array_count += 1


            
            # print("score0", scores[0][0])
            # print("score1", scores[1][0])
            # print("score2", scores[2][0])
            # print("score3", scores[3][0])
            # print("score4", scores[4][0])
            # print("score5", scores[5][0])
            # print("score6", scores[6][0])
            # print("score7", scores[7][0])
            # print("score8", scores[8][0])
            # print("score9", scores[9][0])
            # print("score10", scores[10][0])
            # print("score11", scores[11][0])
            
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
                    if min_part_dtw[s_idx] >= 80 :
                        color = (255, 0, 0)
                    elif min_part_dtw[s_idx] > 60 :
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
                # scores_ = "score " + str(s_idx) + " : " + "{:.2f}".format(scores[s_idx])
                # cv2.putText(image, scores_, (50,50 + (s_idx * 20)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)

            frame_now += 1

            cv2.imshow('Pro-pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

