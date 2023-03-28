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

    key_point_frame = [15, 82, 95, 165]
    scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores_temp = np.zeros(12)
    frame_now = 0
    dtw_array_count = 0
    max_frame_list = []

    scores_list = []

    ins_dtw_info = [[] for i in range(12)]
    stu_dtw_info = [[] for i in range(12)]

    scores = np.zeros(12)
    array = [[0]*2 for j in range(33)]      # (학생)각 랜드마크별 xy좌표 저장 공간
    
    #(공통) 랜드마크 간 연결 리스트
    connects_list = [[11, 12], [14, 12], [13, 11], [16, 14], [15, 13], 
                     [24, 12], [23, 11], [23, 24], [26, 24], [25, 23], [28, 26], [27, 25]]

    max_frame = 0

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
            dtw_how = 10 # ins
            dtw_range = 30 # stu
            
            key_point_range = 5
            
            if(frame_total - frame_now > dtw_how) :
                ins_dtw_info = [[] for i in range(12)]
            
            # frame_info <= 15 인 경우에는 현재 프레임 포함 45프레임을 검사
            # frame_info > 15 인 경우에는 현재 프레임 이전 15프레임 + 이후 30프레임을 10프레임씩 검사
            
            temp = np.zeros(12)
            min_scores = 999    # 해당 프레임의 dtw 최솟값
            ins_min_frames = 0  #dtw가 최소가되는 프레임의 시작값
            stu_min_frames = 0

            
            
            min_part_dtw = np.zeros(12) #dtw 평균이 최소가 되는 프레임의 부위별 dtw값

            # min_dtw = np.zeros(12)
            
            
            if(frame_total - frame_now > dtw_how) :
                print(frame_total)
                print(frame_now)
                for i in range(dtw_array_count, dtw_array_count + dtw_how):
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
            
                if(frame_now < dtw_how):       #현재 프레임이 10 미만인 경우 0 ~ 29프레임까지 비교
                    average = 0
                    for j in range(dtw_range - dtw_how + 1):
                        stu_dtw_info = [[] for i in range(12)]
                        for i in range(dtw_array_count + j, (dtw_array_count + j) + dtw_how):
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
                        for i in range(12):
                            temp[i] = dtw.distance(stu_dtw_info[i], ins_dtw_info[i], window=3)
                        average = np.mean(temp)
                        if(min_scores > average):
                            min_scores = average
                            ins_min_frames = dtw_array_count
                            stu_min_frames = dtw_array_count + j
                            min_part_dtw = (1 - (temp / 20)) * 100
                            # min_dtw = temp
                    
                elif(frame_now >= dtw_how):   #현재 프레임이 10 이상인 경우 (현재 프레임 -10) 부터 (현재 프레임 + 20까지 비교)
                    average = 0
                    for j in range(-5 , dtw_range - dtw_how -5 + 1):
                        stu_dtw_info = [[] for i in range(12)]
                        for i in range(dtw_array_count + j, (dtw_array_count + j) + dtw_how):
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
                        for i in range(12):
                            temp[i] = dtw.distance(stu_dtw_info[i], ins_dtw_info[i], window=3)
                        average = np.mean(temp)
                        if(min_scores > average):
                            min_scores = average
                            ins_min_frames = dtw_array_count
                            stu_min_frames = dtw_array_count + j
                            min_part_dtw = (1 - (temp / 20)) * 100
                            # min_dtw = temp
                    
            
            # print(min_scores)
            # print("ins" + str(ins_min_frames))
            # print("stu" + str(stu_min_frames))

            max_score = 0
            key_point_temp = np.subtract(key_point_frame, key_point_range)    # 키포인트 프레임 리스트에서 key_point_range만큼 뺌 -> 시작 프레임의 값을 지정하기 위해
            
            if(frame_now in key_point_temp):
                average = 0
                for i in range(stu_min_frames, stu_min_frames + dtw_how):
                    scores_temp[0] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][0], ins_info[frame_now + key_point_range][1]]), np.array([stu_info[i][0], stu_info[i][1]]))))
                    scores_temp[1] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][2], ins_info[frame_now + key_point_range][3]]), np.array([stu_info[i][2], stu_info[i][3]]))))
                    scores_temp[2] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][4], ins_info[frame_now + key_point_range][5]]), np.array([stu_info[i][4], stu_info[i][5]]))))
                    scores_temp[3] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][6], ins_info[frame_now + key_point_range][7]]), np.array([stu_info[i][6], stu_info[i][7]]))))
                    scores_temp[4] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][8], ins_info[frame_now + key_point_range][9]]), np.array([stu_info[i][8], stu_info[i][9]]))))
                    scores_temp[5] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][10], ins_info[frame_now + key_point_range][11]]), np.array([stu_info[i][10], stu_info[i][11]]))))
                    scores_temp[6] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][12], ins_info[frame_now + key_point_range][13]]), np.array([stu_info[i][12], stu_info[i][13]]))))
                    scores_temp[7] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][14], ins_info[frame_now + key_point_range][15]]), np.array([stu_info[i][14], stu_info[i][15]]))))
                    scores_temp[8] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][16], ins_info[frame_now + key_point_range][17]]), np.array([stu_info[i][16], stu_info[i][17]]))))
                    scores_temp[9] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][18], ins_info[frame_now + key_point_range][19]]), np.array([stu_info[i][18], stu_info[i][19]]))))
                    scores_temp[10] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][20], ins_info[frame_now + key_point_range][21]]), np.array([stu_info[i][20], stu_info[i][21]]))))
                    scores_temp[11] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][22], ins_info[frame_now + key_point_range][23]]), np.array([stu_info[i][22], stu_info[i][23]]))))
                    average = np.mean(scores_temp)
                    if (average > max_score):
                        max_score = average
                        max_frame = i

                        scores[0] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][0], ins_info[frame_now + key_point_range][1]]), np.array([stu_info[i][0], stu_info[i][1]]))))
                        scores[1] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][2], ins_info[frame_now + key_point_range][3]]), np.array([stu_info[i][2], stu_info[i][3]]))))
                        scores[2] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][4], ins_info[frame_now + key_point_range][5]]), np.array([stu_info[i][4], stu_info[i][5]]))))
                        scores[3] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][6], ins_info[frame_now + key_point_range][7]]), np.array([stu_info[i][6], stu_info[i][7]]))))
                        scores[4] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][8], ins_info[frame_now + key_point_range][9]]), np.array([stu_info[i][8], stu_info[i][9]]))))
                        scores[5] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][10], ins_info[frame_now + key_point_range][11]]), np.array([stu_info[i][10], stu_info[i][11]]))))
                        scores[6] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][12], ins_info[frame_now + key_point_range][13]]), np.array([stu_info[i][12], stu_info[i][13]]))))
                        scores[7] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][14], ins_info[frame_now + key_point_range][15]]), np.array([stu_info[i][14], stu_info[i][15]]))))
                        scores[8] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][16], ins_info[frame_now + key_point_range][17]]), np.array([stu_info[i][16], stu_info[i][17]]))))
                        scores[9] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][18], ins_info[frame_now + key_point_range][19]]), np.array([stu_info[i][18], stu_info[i][19]]))))
                        scores[10] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][20], ins_info[frame_now + key_point_range][21]]), np.array([stu_info[i][20], stu_info[i][21]]))))
                        scores[11] = score(euclid(cos_sim(np.array([ins_info[frame_now + key_point_range][22], ins_info[frame_now + key_point_range][23]]), np.array([stu_info[i][22], stu_info[i][23]]))))

                        print("temp" + str(scores_temp))
                        # print("스코어 : " + str(max_score))
                        # print(max_frame)
                        
                # print(scores)
                print(scores)
                max_frame_list.append(max_frame)
                scores_list.append(list(scores))
                print(max_frame_list)
                print(scores_list)
                print("현재 프레임(ins) : " + str(frame_now))
                print("가장 유사한 프레임(stu) : " + str(max_frame))
                print("스코어 : " + str(max_score))
            

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
            
            # if(frame_now == max_frame) :
            #     for i in range(12):
            #         scores_ = "score " + str(i) + " : " + "{:.2f}".format(scores[i])
            #         cv2.putText(image, scores_, (50,50 + (i * 20)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
                # cv2.imwrite('./keypoint/'+str(frame_now)+'.jpg', image)

            if(key_point_frame[0] < 10 and frame_now < 30):
                cv2.imwrite("./images/frame%d.jpg" % frame_now, image)
            elif(frame_now >= key_point_frame[0] - 10 and frame_now < key_point_frame[0] + 20):
                cv2.imwrite("./images/frame%d.jpg" % frame_now, image)

            if(key_point_frame[1] < 10 and frame_now < 30):
                cv2.imwrite("./images/frame%d.jpg" % frame_now, image)
            elif(frame_now >= key_point_frame[1] - 10 and frame_now < key_point_frame[1] + 20):
                cv2.imwrite("./images/frame%d.jpg" % frame_now, image)

            if(key_point_frame[2] < 10 and frame_now < 30):
                cv2.imwrite("./images/frame%d.jpg" % frame_now, image)
            elif(frame_now >= key_point_frame[2] - 10 and frame_now < key_point_frame[2] + 20):
                cv2.imwrite("./images/frame%d.jpg" % frame_now, image)

            if(key_point_frame[3] < 10 and frame_now < 30):
                cv2.imwrite("./images/frame%d.jpg" % frame_now, image)
            elif(frame_now >= key_point_frame[3] - 10 and frame_now < key_point_frame[3] + 20):
                cv2.imwrite("./images/frame%d.jpg" % frame_now, image)
        

            frame_now += 1
            dtw_array_count += 1
            
            cv2.imshow('Pro-pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    return max_frame_list, scores_list

