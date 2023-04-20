import numpy as np
from numpy.linalg import norm
import math
import cv2
from media_pipe_module import mediapipe_pose
from dtaidistance import dtw
import multiprocessing


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

# multiprocessing - zip
def calculate_distance(args):
    stu, ins, w = args
    return dtw.distance(stu, ins, window=w)


def tracking_dtw(ins_info, stu_info, cap, frame_total, keypoint) :
    pool = multiprocessing.Pool(processes=12)
    #------------------------ (ins) 기준 키프레임
    key_point_frame = keypoint
    #------------------------
    #------------------------ (ins/stu) 점수모음
    scores_temp = np.zeros(12)
    scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores_list = []
    #------------------------
    #------------------------ (ins_info/stu_info -> frame_now번째 프레임에서 -> ins_dtw_info/stu_dtw_info)
    ins_dtw_info = [[] for i in range(12)]
    stu_dtw_info = [[] for i in range(12)]
    #------------------------
    #------------------------ 현재 프레임
    frame_now = 0
    #------------------------
    #------------------------
    max_frame_list = []
    max_frame = 0
    #------------------------
    #------------------------ dtw 탐색 범위
    dtw_how = 10 # ins
    dtw_range = 30 # stu
    #------------------------
    #------------------------ 키포인트 프레임 리스트 - key_point_range -> 시작 프레임의 값 지정
    key_point_range = 5
    #------------------------
    
    #------------------------
    stu_min_frames_list = []

    array = [[0]*2 for i in range(33)]

    #(공통) 랜드마크 간 연결 리스트
    connects_list = [[11, 12], [14, 12], [13, 11], [16, 14], [15, 13], 
                     [24, 12], [23, 11], [23, 24], [26, 24], [25, 23], [28, 26], [27, 25]]

    
    with mediapipe_pose.Pose(
        min_detection_confidence = 0.5,     #사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
        min_tracking_confidence = 0.5,      #포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
        model_complexity = 1,               #포즈 랜드마크 모델의 복합성입니다.
        ) as pose :

        while (cap.isOpened()) :
            success, image = cap.read()

            if not success :
                break
            
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     #영상 높이/넓이 계산
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            image.flags.writeable = False
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
                    land_x, land_y = int(land.x*width), int(land.y*height)

                array[idx][0] = land_x       # 해당 랜드마크의 x좌표입니다.
                array[idx][1] = land_y       # 해당 랜드마크의 y좌표입니다.

            # frame_info <= 15 인 경우에는 현재 프레임 포함 45프레임을 검사
            # frame_info > 15 인 경우에는 현재 프레임 이전 15프레임 + 이후 30프레임을 10프레임씩 검사
            if(frame_total - frame_now > dtw_how) :
                ins_dtw_info = [[] for i in range(12)]
            
            #temp = np.zeros(12)
            min_scores = 999    # 해당 프레임의 dtw 최솟값
            stu_min_frames = 0

            min_part_dtw = np.zeros(12) #dtw 평균이 최소가 되는 프레임의 부위별 dtw값

            # 1.만약 총 프레임 - 현재 프레임이 > 10 보다 크다면(맨 끝 프레임까지 10프레임 넘게 남았다면,)
            # ins_dtw_info에 csv정보들을 추가합니다.
            if(frame_total - frame_now > dtw_how) :
                for i in range(frame_now, frame_now + dtw_how):
                    for j in range(12):
                        ins_dtw_info[j].append(np.array([ins_info[i][j*2], ins_info[i][j*2+1]]))

                # 1-1. 아직 현재프레임이 10을 넘지 않았다면,
                if(frame_now < dtw_how):       
                    average = 0
                    for j in range(dtw_range - dtw_how + 1): #### j가 0~20까지 돌 때
                        stu_dtw_info = [[] for i in range(12)]
                        ####i는 j가 0일 때, 0~9(프레임으로 따지면 1~10)의 벡터값을 저장
                        for i in range(frame_now + j, (frame_now + j) + dtw_how):
                            for k in range(12):
                                stu_dtw_info[k].append(np.array([stu_info[i][2*k], stu_info[i][2*k+1]]))
                        # 평균 dtw 최소거리값
                        temp1 = pool.map(calculate_distance, zip(stu_dtw_info, ins_dtw_info, [3]*12))
                        temp = np.array(temp1)
                        # for i in range(12) :           
                        #     temp[i] = dtw.distance(stu_dtw_info[i], ins_dtw_info[i], window=3)
                        average = np.mean(temp)
                        if(min_scores > average):
                            min_scores = average
                            stu_min_frames = frame_now + j
                            min_part_dtw = (1 - (temp / 20)) * 100
                        

                # 1-2. 현재프레임이 10이상이라면, (현재 프레임 -10) 부터 (현재 프레임 + 20) 까지 비교)
                elif(frame_now >= dtw_how):   
                    average = 0
                    for j in range(-5 , dtw_range - dtw_how -5 + 1):
                        stu_dtw_info = [[] for i in range(12)]
                        for i in range(frame_now + j, (frame_now + j) + dtw_how):
                            for k in range(12):
                                stu_dtw_info[k].append(np.array(stu_info[i][2*k:2*k+2]))


                        temp1 = pool.map(calculate_distance, zip(stu_dtw_info, ins_dtw_info, [3]*12))
                        temp = np.array(temp1)
                        # for i in range(12) :           
                        #     temp[i] = dtw.distance(stu_dtw_info[i], ins_dtw_info[i], window=3, use_pruning=True)
                        average = np.mean(temp)
                        if(min_scores > average):
                            min_scores = average
                            #### 최소값이 평균값으로 갱신될 때의 10개 프레임 중 첫번째 프레임 값
                            stu_min_frames = frame_now + j ####127라인 동일
                            min_part_dtw = (1 - (temp / 20)) * 100
                            
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
                            ins_point = np.array([ins_info[frame_now + key_point_range][j*2], ins_info[frame_now + key_point_range][j*2+1]])
                            stu_point = np.array([stu_info[i][j*2], stu_info[i][j*2+1]])
                            scores[j] = score(ins_point, stu_point)


                max_frame_list.append(max_frame)
                scores_list.append(list(scores))
                


        #cv2 - 랜드마크 선 표현
            for s_idx, i in enumerate(connects_list) :
                if array[i[0]][0] is not None and array[i[0]][1] is not None and array[i[1]][0] is not None and array[i[1]][1] is not None:
                    if scores[s_idx] >= 85 :
                        color = (255, 255, 0)
                    elif scores[s_idx] > 70 :
                        color = (0, 255, 255)
                    else:
                        color = (255, 153, 255)
                    cv2.line(
                    image,
                    (array[i[0]][0], array[i[0]][1]),
                    (array[i[1]][0], array[i[1]][1]),
                    color = color,
                    thickness = 2
                    )
            
            # if(frame_now == max_frame) :
            #     for i in range(12):
            #         scores_ = "score " + str(i) + " : " + "{:.2f}".format(scores[i])
            #         cv2.putText(image, scores_, (50,50 + (i * 20)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
                # cv2.imwrite('./keypoint/'+str(frame_now)+'.jpg', image)

            for key_point in key_point_frame:
                if key_point < 10 and frame_now < 30:
                    cv2.imwrite("./images/frame{:03d}.jpg".format(frame_now), image)
                elif frame_now >= key_point - 10 and frame_now < key_point + 20:
                    cv2.imwrite("./images/frame{:03d}.jpg".format(frame_now), image)

            frame_now += 1
            
            # cv2.imshow('Pro-pose', image)
            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break
    pool.close()
    pool.join()
    return max_frame_list, scores_list



