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


def distance(a,b) :
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def pose_drawing(ins_info, stu_info, ins, stu, sort) :

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
            image2_1 = cv2.rectangle(image2_1, (0,0), (1280,720), (255,255,255), -1)
            height, width, _ = image1.shape

            result1 = pose.process(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            result2 = pose.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

            landmarks1 = result1.pose_landmarks.landmark
            landmarks2 = result2.pose_landmarks.landmark


            x_coords1 = [landmark.x for landmark in landmarks1]
            y_coords1 = [landmark.y for landmark in landmarks1]

            mean_x1 = sum(x_coords1) / len(x_coords1)
            mean_y1 = sum(y_coords1) / len(y_coords1)
            ins_11 = [int(x_coords1[11] * width), int(y_coords1[11] * height)]
            ins_13 = [int(x_coords1[13] * width), int(y_coords1[13] * height)]
            ins_15 = [int(x_coords1[15] * width), int(y_coords1[15] * height)]

            ins_12 = [int(x_coords1[12] * width), int(y_coords1[12] * height)]
            ins_14 = [int(x_coords1[14] * width), int(y_coords1[14] * height)]
            ins_16 = [int(x_coords1[16] * width), int(y_coords1[16] * height)]

            ins_23 = [int(x_coords1[23] * width), int(y_coords1[23] * height)]
            ins_25 = [int(x_coords1[25] * width), int(y_coords1[25] * height)]
            ins_27 = [int(x_coords1[27] * width), int(y_coords1[27] * height)]

            ins_24 = [int(x_coords1[24] * width), int(y_coords1[24] * height)]
            ins_26 = [int(x_coords1[26] * width), int(y_coords1[26] * height)]
            ins_28 = [int(x_coords1[28] * width), int(y_coords1[28] * height)]






            x_coords2 = [landmark.x for landmark in landmarks2]
            y_coords2 = [landmark.y for landmark in landmarks2]

            mean_x2 = sum(x_coords2) / len(x_coords2)
            mean_y2 = sum(y_coords2) / len(y_coords2)
            stu_11 = [int(x_coords2[11] * width), int(y_coords2[11] * height)]
            stu_13 = [int(x_coords2[13] * width), int(y_coords2[13] * height)]
            stu_15 = [int(x_coords2[15] * width), int(y_coords2[15] * height)]

            stu_12 = [int(x_coords2[12] * width), int(y_coords2[12] * height)]
            stu_14 = [int(x_coords2[14] * width), int(y_coords2[14] * height)]
            stu_16 = [int(x_coords2[16] * width), int(y_coords2[16] * height)]

            stu_23 = [int(x_coords2[23] * width), int(y_coords2[23] * height)]
            stu_25 = [int(x_coords2[25] * width), int(y_coords2[25] * height)]
            stu_27 = [int(x_coords2[27] * width), int(y_coords2[27] * height)]

            stu_24 = [int(x_coords2[24] * width), int(y_coords2[24] * height)]
            stu_26 = [int(x_coords2[26] * width), int(y_coords2[26] * height)]
            stu_28 = [int(x_coords2[28] * width), int(y_coords2[28] * height)]

            center1 = (int(mean_x1 * width), int(mean_y1 * height))
            center2 = (int(mean_x2 * width), int(mean_y2 * height))

            ins_distance = int((distance(ins_11, center1) + distance(ins_12, center1) + distance(ins_23, center1) + distance(ins_24, center1)) / 4)
            stu_distance = int((distance(stu_11, center2) + distance(stu_12, center2) + distance(stu_23, center2) + distance(stu_24, center2)) / 4)

            mag = stu_distance / ins_distance
            mag = round(mag,1)


            center1 = (int(center1[0] * mag), int(center1[1] * mag))
            landmarks_shifted = (center2[0] - center1[0], center2[1] - center1[1])

            
            add_x = stu_11[0]-ins_11[0]
            add_y = stu_11[1]-ins_11[1]
            ins_11 = (ins_11[0] + add_x, ins_11[1] + add_y)
            ins_13 = (ins_13[0] + add_x, ins_13[1] + add_y)
            ins_15 = (ins_15[0] + add_x, ins_15[1] + add_y)

            add_x = stu_12[0]-ins_12[0]
            add_y = stu_12[1]-ins_12[1]
            ins_12 = (ins_12[0] + add_x, ins_12[1] + add_y)
            ins_14 = (ins_14[0] + add_x, ins_14[1] + add_y)
            ins_16 = (ins_16[0] + add_x, ins_16[1] + add_y)

            add_x = stu_23[0]-ins_23[0]
            add_y = stu_23[1]-ins_23[1]
            ins_23 = (ins_23[0] + add_x, ins_23[1] + add_y)
            ins_25 = (ins_25[0] + add_x, ins_25[1] + add_y)
            ins_27 = (ins_27[0] + add_x, ins_27[1] + add_y)

            add_x = stu_24[0]-ins_24[0]
            add_y = stu_24[1]-ins_24[1]
            ins_24 = (ins_24[0] + add_x, ins_24[1] + add_y)
            ins_26 = (ins_26[0] + add_x, ins_26[1] + add_y)
            ins_28 = (ins_28[0] + add_x, ins_28[1] + add_y)

            cv2.line(
                image2,
                ins_11,
                ins_13,
                color = (255,0,0),
                thickness = 5
            )

            cv2.line(
                image2,
                ins_13,
                ins_15,
                color = (255,0,0),
                thickness = 5
            )


            cv2.line(
                image2,
                ins_12,
                ins_14,
                color = (255,0,0),
                thickness = 5
            )

            cv2.line(
                image2,
                ins_14,
                ins_16,
                color = (255,0,0),
                thickness = 5
            )


            cv2.line(
                image2,
                ins_23,
                ins_25,
                color = (255,0,0),
                thickness = 5
            )
            cv2.line(
                image2,
                ins_25,
                ins_27,
                color = (255,0,0),
                thickness = 5
            )


            cv2.line(
                image2,
                ins_24,
                ins_26,
                color = (255,0,0),
                thickness = 5
            )
            cv2.line(
                image2,
                ins_26,
                ins_28,
                color = (255,0,0),
                thickness = 5
            )

            



            cv2.line(
                image2,
                ins_11,
                ins_12,
                color = (255,0,0),
                thickness = 5
            )
            cv2.line(
                image2,
                ins_23,
                ins_24,
                color = (255,0,0),
                thickness = 5
            )
            cv2.line(
                image2,
                ins_11,
                ins_23,
                color = (255,0,0),
                thickness = 5
            )
            cv2.line(
                image2,
                ins_12,
                ins_24,
                color = (255,0,0),
                thickness = 5
            )

            result = cv2.addWeighted(image2, 0.75, image2_1, 0.25, 0)
            cv2.imwrite("/Users/jangjun-yeong/webpage/public/" + sort + "/" + str(idx) + ".jpg", result)

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