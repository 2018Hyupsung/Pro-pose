from numpy.linalg import norm
import math
import numpy as np
import cv2
from media_pipe_module import mediapipe_pose

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



def pose_drawing(ins_info, stu_info, ins, stu) :

    scores = [[] for i in range(12)]
    cos = [[] for i in range(12)]

    ins_frames = []
    stu_frames = []
    for i in ins :
        ins_frames.append(int(i[18:21]))
    for i in stu :
        stu_frames.append(int(i[14:17]))
        
    idx = 0

    # 랜드마크 간 연결 리스트
    connects_list = [[11, 12], [14, 12], [13, 11], [16, 14], [15, 13], 
                     [24, 12], [23, 11], [23, 24], [26, 24], [25, 23], 
                     [28, 26], [27, 25]]



    with mediapipe_pose.Pose(
        static_image_mode = True,
        min_detection_confidence = 0.5,     # 사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
        min_tracking_confidence = 0.5,      # 포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
        model_complexity = 1,               # 포즈 랜드마크 모델의 복합성입니다.
        ) as pose :
       

       
        while (idx < len(ins)) :

            for i in range(12) :
                ins_point = np.array([ins_info[ins_frames[idx]][i*2], ins_info[ins_frames[idx]][i*2+1]])
                stu_point = np.array([stu_info[stu_frames[idx]][i*2], stu_info[stu_frames[idx]][i*2+1]])
                scores[i] = score(ins_point, stu_point)
                cos[i] = cos_sim(ins_point, stu_point)


            image1 = cv2.imread(ins[idx])
            image1_1 = cv2.imread(ins[idx])
            image2 = cv2.imread(stu[idx])
            image2_1 = cv2.imread(stu[idx])
            height, width, _ = image1.shape

            result1 = pose.process(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            result2 = pose.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

            landmarks1 = result1.pose_landmarks.landmark
            landmarks2 = result2.pose_landmarks.landmark


            x_coords1 = [landmark.x for landmark in landmarks1]
            y_coords1 = [landmark.y for landmark in landmarks1]

            mean_x1 = sum(x_coords1) / len(x_coords1)
            mean_y1 = sum(y_coords1) / len(y_coords1)

            center1 = (int(mean_x1 * width), int(mean_y1 * height))


            x_coords2 = [landmark.x for landmark in landmarks2]
            y_coords2 = [landmark.y for landmark in landmarks2]

            mean_x2 = sum(x_coords2) / len(x_coords2)
            mean_y2 = sum(y_coords2) / len(y_coords2)

            center2 = (int(mean_x2 * width), int(mean_y2 * height))

            landmarks_shifted = (center2[0] - center1[0], center2[1] - center1[1])

            for i in range (len(connects_list)) :
                # if(scores[i] >= 85) :
                #     continue
                color = (255,0,0)
                coord1 = (int(result1.pose_landmarks.landmark[connects_list[i][0]].x * width), int(result1.pose_landmarks.landmark[connects_list[i][0]].y * height))
                coord2 = (int(result1.pose_landmarks.landmark[connects_list[i][1]].x * width), int(result1.pose_landmarks.landmark[connects_list[i][1]].y * height))
                coord1_1 = (coord1[0] + landmarks_shifted[0], coord1[1] + landmarks_shifted[1])
                coord2_1 = (coord2[0] + landmarks_shifted[0], coord2[1] + landmarks_shifted[1])

                cv2.line(
                    image2,
                    coord1_1,
                    coord2_1,
                    color = color,
                    thickness = 5
                )

            result = cv2.addWeighted(image2, 0.7, image2_1, 0.3, 0)
            cv2.imwrite("./final/" + str(idx) + ".jpg", result)

            idx += 1






# #교수자
# coord1 = (int(result1.pose_landmarks.landmark[connects_list[i][0]].x * width), int(result1.pose_landmarks.landmark[connects_list[i][0]].y * height))
# coord2 = (int(result1.pose_landmarks.landmark[connects_list[i][1]].x * width), int(result1.pose_landmarks.landmark[connects_list[i][1]].y * height))
# #학습자
# coord3 = (int(result2.pose_landmarks.landmark[connects_list[i][0]].x * width), int(result2.pose_landmarks.landmark[connects_list[i][0]].y * height))
# coord4 = (int(result2.pose_landmarks.landmark[connects_list[i][1]].x * width), int(result2.pose_landmarks.landmark[connects_list[i][1]].y * height))


# # moved_x1 = coord3[0] - coord2[0]
# # moved_y1 = coord3[1] - coord2[1]
# # coord1 = ((coord1[0] + moved_x1),(coord1[1] + moved_y1))
# # coord2 = ((coord2[0] + moved_x1),(coord2[1] + moved_y1))