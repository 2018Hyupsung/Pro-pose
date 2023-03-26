
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







def tracking(ins_info, stu_info, cap, frame_total) :

    key_point_frame = [15, 82, 95, 165]
    scores_temp = np.zeros(12)
    frame_now = 0
    dtw_array_count = 0
    max_frame_list = []
    scores_list = []
    ins_dtw_info = [[] for i in range(12)]
    stu_dtw_info = [[] for i in range(12)]
    scores = np.zeros(12)


    scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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


    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     #영상 높이/넓이 계산
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    

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
                    for j in range(12):
                        ins_dtw_info[j].append(np.array([ins_info[i][j*2], ins_info[i][j*2+1]]))

            
                if(frame_now < dtw_how):       #현재 프레임이 10 미만인 경우 0 ~ 29프레임까지 비교
                    average = 0
                    for j in range(dtw_range - dtw_how + 1):
                        stu_dtw_info = [[] for i in range(12)]
                        for i in range(dtw_array_count + j, (dtw_array_count + j) + dtw_how):
                            for k in range(12):
                                stu_dtw_info[k].append(np.array([stu_info[i][2*k], stu_info[i][2*k+1]]))

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
                            for k in range(12):
                                stu_dtw_info[k].append(np.array(stu_info[i][2*k:2*k+2]))

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
                    for j in range(12):
                        ins_point = np.array([ins_info[frame_now + key_point_range][j*2], ins_info[frame_now + key_point_range][j*2+1]])
                        stu_point = np.array([stu_info[i][j*2], stu_info[i][j*2+1]])
                        scores_temp[j] = score(euclid(cos_sim(ins_point, stu_point)))

                    average = np.mean(scores_temp)
                    if (average > max_score):
                        max_score = average
                        max_frame = i

                        for j in range(12):
                            inst_keypoints = np.array(ins_info[frame_now + key_point_range][j*2:j*2+2])
                            stu_keypoints = np.array(stu_info[i][j*2:j*2+2])
                            scores[j] = score(euclid(cos_sim(inst_keypoints, stu_keypoints)))


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

            for key_point in key_point_frame:
                if key_point < 10 and frame_now < 30:
                    cv2.imwrite("./images/frame%d.jpg" % frame_now, image)
                elif frame_now >= key_point - 10 and frame_now < key_point + 20:
                    cv2.imwrite("./images/frame%d.jpg" % frame_now, image)

            frame_now += 1
            dtw_array_count += 1
            
            cv2.imshow('Pro-pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    return max_frame_list, scores_list

