
import numpy as np
from numpy.linalg import norm
import math
import cv2
from media_pipe_module import mediapipe_pose
from dtaidistance import dtw
font_italic = "FONT_ITALIC"





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



# score
def score(a, b):
    cos = cos_sim(a, b)
    euc = euclid(cos)
    if np.isnan(euc):
        return np.nan
    return 100 - (50 * euc)



def tracking(ins_info, stu_info, cap, frame_total) :

    key_point_frame = [15, 82, 95, 165]
    scores_temp = np.zeros(12)
    frame_now = 0
    max_frame_list = []
    scores_list = []
    ins_dtw_info = [[] for i in range(12)]
    stu_dtw_info = [[] for i in range(12)]



    scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    #(공통) 랜드마크 간 연결 리스트
    connects_list = [[11, 12], [14, 12], [13, 11], [16, 14], [15, 13], 
                     [24, 12], [23, 11], [23, 24], [26, 24], [25, 23], 
                     [28, 26], [27, 25]]

    max_frame = 0


    

    with mediapipe_pose.Pose(
        min_detection_confidence = 0.5,     #사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
        min_tracking_confidence = 0.5,      #포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
        model_complexity = 1,               #포즈 랜드마크 모델의 복합성입니다.
        ) as pose :

        while (cap.isOpened()) :
            success, image = cap.read()

            if not success :
                break
            

            image.flags.writeable = False
            results = pose.process(image)
            
            
            #stu의 dtw를 10프레임을 고정으로 하고 비교한다.
            #ins의 dtw를 30프레임 범위에서 비교하도록 한다.
            #15fps  현재 30프레임을 10프레임씩 총 21번 비교
            dtw_how = 10 # ins
            dtw_range = 30 # stu
            
            key_point_range = 5
            
            if(frame_total - frame_now > dtw_how) :
                ins_dtw_info = [[] for i in range(12)]
            
            # frame_info <= 15 인 경우에는 현재 프레임 포함 45프레임을 검사
            # frame_info > 15 인 경우에는 현재 프레임 이전 15프레임 + 이후 30프레임을 10프레임씩 검사
            
            temp = np.zeros(12)
            min_scores = 999    # 해당 프레임의 dtw 최솟값
            stu_min_frames = 0

            
            # 1.만약 총 프레임 - 현재 프레임이 > 10 보다 크다면(맨 끝 프레임까지 10프레임 넘게 남았다면,)
            # ins_dtw_info에 csv정보들을 추가합니다.
            if(frame_total - frame_now > dtw_how) :
                for i in range(frame_now, frame_now + dtw_how):
                    for j in range(12):
                        ins_dtw_info[j].append(np.array([ins_info[i][j*2], ins_info[i][j*2+1]]))

                # 1-1. 아직 현재프레임이 10을 넘지 않았다면, 0 ~ 29프레임까지 비교
                if(frame_now < dtw_how):       
                    average = 0
                    for j in range(dtw_range - dtw_how + 1): # 21
                        for i in range(frame_now + j, (frame_now + j) + dtw_how): # 현재+
                            for k in range(12):
                                stu_dtw_info[k].append(np.array([stu_info[i][2*k], stu_info[i][2*k+1]]))

                        for i in range(12):
                            temp[i] = dtw.distance(stu_dtw_info[i], ins_dtw_info[i], window=3)
                        average = np.mean(temp)
                        if(min_scores > average):
                            min_scores = average
                            stu_min_frames = frame_now + j


                    
                elif(frame_now >= dtw_how):   #현재 프레임이 10 이상인 경우 (현재 프레임 -10) 부터 (현재 프레임 + 20까지 비교)
                    average = 0
                    for j in range(-5 , dtw_range - dtw_how -5 + 1):
                        stu_dtw_info = [[] for i in range(12)]
                        for i in range(frame_now + j, (frame_now + j) + dtw_how):
                            for k in range(12):
                                stu_dtw_info[k].append(np.array(stu_info[i][2*k:2*k+2]))

                        for i in range(12):
                            temp[i] = dtw.distance(stu_dtw_info[i], ins_dtw_info[i], window=3)
                        average = np.mean(temp)
                        if(min_scores > average):
                            min_scores = average
                            stu_min_frames = frame_now + j


            max_score = 0
            key_point_temp = np.subtract(key_point_frame, key_point_range)    # 키포인트 프레임 리스트에서 key_point_range만큼 뺌 -> 시작 프레임의 값을 지정하기 위해
            
            if(frame_now in key_point_temp):
                average = 0
                for i in range(stu_min_frames, stu_min_frames + dtw_how):
                    for j in range(12):
                        ins_point = np.array([ins_info[frame_now + key_point_range][j*2], ins_info[frame_now + key_point_range][j*2+1]])
                        stu_point = np.array([stu_info[i][j*2], stu_info[i][j*2+1]])
                        scores_temp[j] = score(ins_point, stu_point)

                    average = np.mean(scores_temp)
                    if (average > max_score):
                        max_score = average
                        max_frame = i

                        for j in range(12):
                            scores[j] = score(ins_point, stu_point)


                        print("temp" + str(scores_temp))

                max_frame_list.append(max_frame)
                scores_list.append(list(scores))
                
            for key_point in key_point_frame:
                if key_point < 10 and frame_now < 30:
                    cv2.imwrite("./images/frame%d.jpg" % frame_now, image)
                elif frame_now >= key_point - 10 and frame_now < key_point + 20:
                    cv2.imwrite("./images/frame%d.jpg" % frame_now, image)

            frame_now += 1
            
            cv2.imshow('Pro-pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    return max_frame_list, scores_list

