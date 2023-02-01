from cgitb import reset
from re import T
from sre_constants import SUCCESS
from tkinter import Image
from unittest import result
import cv2
import mediapipe as mp
import time
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from dtaidistance import dtw_ndim
from numpy import append, dot
import numpy as np

Red_Line = 75
Orange_Line = 85
 
#L2 정규화
def L2(inp):
    result = 0
    for i in inp:
        result += i ** 2
    return np.sqrt(result)

#코사인 유사도
def cos_sim(inp1, inp2):
    return dot(inp1, inp2)/(L2(inp1) * L2(inp2))

#점수화
def score(inp1,inp2):

    cs = 2*(1-cos_sim(inp1,inp2))
    if cs <0:
        cs= -cs
    return 100-(50*(np.sqrt(cs)))

#시범자 부위별 데이터를 담을 리스트 생성
video_pose_data = []
video_i_pose_data = []


#참여자 부위별 데이터를 담을 리스트 생성
cam_pose_data = []
cam_i_pose_data = []

#참여자 부위별 데이터를 담을 리스트 생성(좌우반전)
cam_reverse_pose_data = []
cam_i_reverse_pose_data = []


mpDraw = mp.solutions.drawing_utils
mpDraw2 = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mpPose2 = mp.solutions.pose
pose=mpPose.Pose()
pose2=mpPose2.Pose()

cap = cv2.VideoCapture('PoseVideos/solo6_Trim.mp4')

my_pose='PoseVideos/solo5_Trim.mp4'
capture = cv2.VideoCapture(my_pose)
capture_reverse = cv2.VideoCapture(my_pose)
result_cam=cv2.VideoCapture(my_pose)

while True:
    SUCCESS, img = cap.read()
    if  img is None:
         break
    
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks
                              ,mpPose.POSE_CONNECTIONS)
        
    #시범자의 각 부위별 좌표를 저장
    try:
        poseData= []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            data = np.array([lm.x, lm.y])
            poseData.append(data)
        video_i_pose_data.append(np.array(poseData[11]-poseData[12]))#양쪽 어깨 11 to 12 (i=0)
        video_i_pose_data.append(np.array(poseData[12]-poseData[14]))#왼쪽 윗팔 12 to 14 (i=1)
        video_i_pose_data.append(np.array(poseData[11]-poseData[13]))#오른쪽 윗팔 11 to 13 (i=2)
        video_i_pose_data.append(np.array(poseData[14]-poseData[16]))#왼쪽 아래팔 14 to 16 (i=3)
        video_i_pose_data.append(np.array(poseData[13]-poseData[15]))#오른쪽 아래팔 13 to 15 (i=4)
        video_i_pose_data.append(np.array(poseData[12]-poseData[24]))#왼쪽 옆구리 12 to 24 (i=5)
        video_i_pose_data.append(np.array(poseData[11]-poseData[23]))#오른쪽 옆구리 11 to 23 (i=6)
        video_i_pose_data.append(np.array(poseData[23]-poseData[24]))#양쪽 골반 23 to 24 (i=7)
        video_i_pose_data.append(np.array(poseData[24]-poseData[26]))#왼쪽 윗다리 24 to 26  (i=8)
        video_i_pose_data.append(np.array(poseData[23]-poseData[25]))#오른쪽 윗다리 23 to 25 (i=9)
        video_i_pose_data.append(np.array(poseData[26]-poseData[28]))#왼쪽 아래다리 26 to 28 (i=10)
        video_i_pose_data.append(np.array(poseData[25]-poseData[27]))#오른쪽 아래다리 25 to 27 (i=11)

        video_pose_data.append(np.array(video_i_pose_data))
        video_i_pose_data.clear()
        poseData.clear()
        
    #예외처리(시범자) 검출되지 않을 경우 0,0       
    except (TypeError, AttributeError):
        video_i_pose_data.append(np.array([0,0]))#양쪽 어깨 11 to 12 (i=0)
        video_i_pose_data.append(np.array([0,0]))#왼쪽 윗팔 12 to 14 (i=1)
        video_i_pose_data.append(np.array([0,0]))#오른쪽 윗팔 11 to 13 (i=2)
        video_i_pose_data.append(np.array([0,0]))#왼쪽 아래팔 14 to 16 (i=3)
        video_i_pose_data.append(np.array([0,0]))#오른쪽 아래팔 13 to 15 (i=4)
        video_i_pose_data.append(np.array([0,0]))#왼쪽 옆구리 12 to 24 (i=5)
        video_i_pose_data.append(np.array([0,0]))#오른쪽 옆구리 11 to 23 (i=6)
        video_i_pose_data.append(np.array([0,0]))#양쪽 골반 23 to 24 (i=7)
        video_i_pose_data.append(np.array([0,0]))#왼쪽 윗다리 24 to 26  (i=8)
        video_i_pose_data.append(np.array([0,0]))#오른쪽 윗다리 23 to 25 (i=9)
        video_i_pose_data.append(np.array([0,0]))#왼쪽 아래다리 26 to 28 (i=10)
        video_i_pose_data.append(np.array([0,0]))#오른쪽 아래다리 25 to 27 (i=11)

        video_pose_data.append(np.array(video_i_pose_data))
        video_i_pose_data.clear()
        poseData.clear()
   

#참여자 프레임 가져오기
while True:
    SUCCESS, me = capture.read()
    if  me is None:
         break
    
    imgRGB2=cv2.cvtColor(me,cv2.COLOR_BGR2RGB)
    results2=pose2.process(imgRGB2)
    #print(results.pose_landmarks)
    if results2.pose_landmarks:
        mpDraw2.draw_landmarks(me,results2.pose_landmarks
                              ,mpPose2.POSE_CONNECTIONS)  
    #참여자의 각 부위별 좌표를 저장
    try:
        poseData= []
        for id, lm in enumerate(results2.pose_landmarks.landmark):
            h, w, c = me.shape
            data = np.array([lm.x, lm.y])
            poseData.append(data)
        cam_i_pose_data.append(np.array(poseData[11]-poseData[12]))# 양쪽 어깨 11 to 12 (i=0)
        cam_i_pose_data.append(np.array(poseData[12]-poseData[14]))# 왼쪽 윗팔 12 to 14 (i=1)
        cam_i_pose_data.append(np.array(poseData[11]-poseData[13]))#오른쪽 윗팔 11 to 13 (i=2)
        cam_i_pose_data.append(np.array(poseData[14]-poseData[16]))#왼쪽 아래팔 14 to 16 (i=3)
        cam_i_pose_data.append(np.array(poseData[13]-poseData[15]))#오른쪽 아래팔 13 to 15 (i=4)
        cam_i_pose_data.append(np.array(poseData[12]-poseData[24]))#왼쪽 옆구리 12 to 24 (i=5)
        cam_i_pose_data.append(np.array(poseData[11]-poseData[23]))#오른쪽 옆구리 11 to 23 (i=6)
        cam_i_pose_data.append(np.array(poseData[23]-poseData[24]))#양쪽 골반 23 to 24 (i=7)
        cam_i_pose_data.append(np.array(poseData[24]-poseData[26]))#왼쪽 윗다리 24 to 26  (i=8)
        cam_i_pose_data.append(np.array(poseData[23]-poseData[25]))#오른쪽 윗다리 23 to 25 (i=9)
        cam_i_pose_data.append(np.array(poseData[26]-poseData[28]))#왼쪽 아래다리 26 to 28 (i=10)
        cam_i_pose_data.append(np.array(poseData[25]-poseData[27]))#오른쪽 아래다리 25 to 27 (i=11)
        
        cam_pose_data.append(np.array(cam_i_pose_data))
        cam_i_pose_data.clear()
        poseData.clear()   
                
    #예외처리(참여자) 검출되지 않을 경우 0,0       
    except (TypeError, AttributeError):
        cam_i_pose_data.append(np.array([0,0]))# 양쪽 어깨 11 to 12 (i=0)
        cam_i_pose_data.append(np.array([0,0]))# 왼쪽 윗팔 12 to 14 (i=1)
        cam_i_pose_data.append(np.array([0,0]))#오른쪽 윗팔 11 to 13 (i=2)
        cam_i_pose_data.append(np.array([0,0]))#왼쪽 아래팔 14 to 16 (i=3)
        cam_i_pose_data.append(np.array([0,0]))#오른쪽 아래팔 13 to 15 (i=4)
        cam_i_pose_data.append(np.array([0,0]))#왼쪽 옆구리 12 to 24 (i=5)
        cam_i_pose_data.append(np.array([0,0]))#오른쪽 옆구리 11 to 23 (i=6)
        cam_i_pose_data.append(np.array([0,0]))#양쪽 골반 23 to 24 (i=7)
        cam_i_pose_data.append(np.array([0,0]))#왼쪽 윗다리 24 to 26  (i=8)
        cam_i_pose_data.append(np.array([0,0]))#오른쪽 윗다리 23 to 25 (i=9)
        cam_i_pose_data.append(np.array([0,0]))#왼쪽 아래다리 26 to 28 (i=10)
        cam_i_pose_data.append(np.array([0,0]))#오른쪽 아래다리 25 to 27 (i=11)
        
        cam_pose_data.append(np.array(cam_i_pose_data))
        cam_i_pose_data.clear()
        poseData.clear() 
        

#참여자 프레임 가져오기(좌우 반전)
while True:
    SUCCESS, me = capture_reverse.read()
    if  me is None:
         break
    me = cv2.flip(me,1) #좌우반전
    imgRGB2=cv2.cvtColor(me,cv2.COLOR_BGR2RGB)
    results2=pose2.process(imgRGB2)
    #print(results.pose_landmarks)
    if results2.pose_landmarks:
        mpDraw2.draw_landmarks(me,results2.pose_landmarks
                              ,mpPose2.POSE_CONNECTIONS)  
    #참여자의 각 부위별 좌표를 저장
    try:
        poseData= []
        for id, lm in enumerate(results2.pose_landmarks.landmark):
            h, w, c = me.shape
            data = np.array([lm.x, lm.y])
            poseData.append(data)
        cam_i_reverse_pose_data.append(np.array(poseData[11]-poseData[12]))# 양쪽 어깨 11 to 12 (i=0)
        cam_i_reverse_pose_data.append(np.array(poseData[12]-poseData[14]))# 왼쪽 윗팔 12 to 14 (i=1)
        cam_i_reverse_pose_data.append(np.array(poseData[11]-poseData[13]))#오른쪽 윗팔 11 to 13 (i=2)
        cam_i_reverse_pose_data.append(np.array(poseData[14]-poseData[16]))#왼쪽 아래팔 14 to 16 (i=3)
        cam_i_reverse_pose_data.append(np.array(poseData[13]-poseData[15]))#오른쪽 아래팔 13 to 15 (i=4)
        cam_i_reverse_pose_data.append(np.array(poseData[12]-poseData[24]))#왼쪽 옆구리 12 to 24 (i=5)
        cam_i_reverse_pose_data.append(np.array(poseData[11]-poseData[23]))#오른쪽 옆구리 11 to 23 (i=6)
        cam_i_reverse_pose_data.append(np.array(poseData[23]-poseData[24]))#양쪽 골반 23 to 24 (i=7)
        cam_i_reverse_pose_data.append(np.array(poseData[24]-poseData[26]))#왼쪽 윗다리 24 to 26  (i=8)
        cam_i_reverse_pose_data.append(np.array(poseData[23]-poseData[25]))#오른쪽 윗다리 23 to 25 (i=9)
        cam_i_reverse_pose_data.append(np.array(poseData[26]-poseData[28]))#왼쪽 아래다리 26 to 28 (i=10)
        cam_i_reverse_pose_data.append(np.array(poseData[25]-poseData[27]))#오른쪽 아래다리 25 to 27 (i=11)
        
        cam_reverse_pose_data.append(np.array(cam_i_reverse_pose_data))
        cam_i_reverse_pose_data.clear()
        poseData.clear()   
                
    #예외처리(참여자) 검출되지 않을 경우 0,0       
    except (TypeError, AttributeError):
        cam_i_reverse_pose_data.append(np.array([0,0]))# 양쪽 어깨 11 to 12 (i=0)
        cam_i_reverse_pose_data.append(np.array([0,0]))# 왼쪽 윗팔 12 to 14 (i=1)
        cam_i_reverse_pose_data.append(np.array([0,0]))#오른쪽 윗팔 11 to 13 (i=2)
        cam_i_reverse_pose_data.append(np.array([0,0]))#왼쪽 아래팔 14 to 16 (i=3)
        cam_i_reverse_pose_data.append(np.array([0,0]))#오른쪽 아래팔 13 to 15 (i=4)
        cam_i_reverse_pose_data.append(np.array([0,0]))#왼쪽 옆구리 12 to 24 (i=5)
        cam_i_reverse_pose_data.append(np.array([0,0]))#오른쪽 옆구리 11 to 23 (i=6)
        cam_i_reverse_pose_data.append(np.array([0,0]))#양쪽 골반 23 to 24 (i=7)
        cam_i_reverse_pose_data.append(np.array([0,0]))#왼쪽 윗다리 24 to 26  (i=8)
        cam_i_reverse_pose_data.append(np.array([0,0]))#오른쪽 윗다리 23 to 25 (i=9)
        cam_i_reverse_pose_data.append(np.array([0,0]))#왼쪽 아래다리 26 to 28 (i=10)
        cam_i_reverse_pose_data.append(np.array([0,0]))#오른쪽 아래다리 25 to 27 (i=11)
        
        cam_reverse_pose_data.append(np.array(cam_i_reverse_pose_data))
        cam_i_reverse_pose_data.clear()
        poseData.clear()  
        
video_pose_data_L2=[]
video_pose_i_data_L2=[]       
for i in range(len(video_pose_data)):
    for j in range (len(video_pose_data[i])):
        if video_pose_data[i][j][0]==0 :
            video_pose_i_data_L2.append(np.array([0,0]))
        else:
            video_pose_i_data_L2.append(np.array(video_pose_data[i][j]/L2(video_pose_data[i][j])))
    video_pose_data_L2.append(np.array(video_pose_i_data_L2))
    video_pose_i_data_L2.clear()

cam_pose_data_L2=[]
cam_pose_i_data_L2=[]       
for i in range(len(cam_pose_data)):
    for j in range (len(cam_pose_data[i])):
        if cam_pose_data[i][j][0]==0 :
            video_pose_i_data_L2.append(np.array([0,0]))
        else:
            cam_pose_i_data_L2.append(np.array(cam_pose_data[i][j]/L2(cam_pose_data[i][j])))
    cam_pose_data_L2.append(np.array(cam_pose_i_data_L2))
    cam_pose_i_data_L2.clear()

cam_reverse_pose_data_L2=[]
cam_reverse_pose_i_data_L2=[]       
for i in range(len(cam_reverse_pose_data)):
    for j in range (len(cam_reverse_pose_data[i])):
        if cam_reverse_pose_data[i][j][0]==0:
            cam_reverse_pose_i_data_L2.append(np.array([0,0]))
        else:
            cam_reverse_pose_i_data_L2.append(np.array(cam_reverse_pose_data[i][j]/L2(cam_reverse_pose_data[i][j])))
    cam_reverse_pose_data_L2.append(np.array(cam_reverse_pose_i_data_L2))
    cam_reverse_pose_i_data_L2.clear()
           
    
d1 = dtw_ndim.distance(cam_pose_data_L2, video_pose_data_L2)
d2 = dtw_ndim.distance(cam_reverse_pose_data_L2, video_pose_data_L2)
if d1 > d2 :
    cam_pose_data=cam_reverse_pose_data
    
    
path = dtw_ndim.warping_path(cam_pose_data_L2, video_pose_data_L2)

    
score_list=[]
score_i_list=[] 
for i in range(len(path)):
    score_i_list.append(path[i][0])
    #print(path[i][0],"번째 프레임")
    sum=0
    if cam_pose_data[path[i][0]][0][0]==0 or video_pose_data[path[i][1]][0][0]==0:
        #print("검출되지않음")
        for k in range(13):
            score_i_list.append(0)
    else:
        for m in range(12):
            if m ==0:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("양쪽어깨 유사도: ",sim_score,"%")                
                sum+=sim_score
            elif m ==1:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("왼쪽윗팔 유사도: ",sim_score,"%")                
                sum+=sim_score
            elif m ==2:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("오른쪽윗팔 유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==3:
                sim_score=score(cam_pose_data[path[path[i][0]][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("왼쪽아랫팔 유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==4:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("오른쪽 아랫팔 유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==5:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("왼쪽 옆구리 유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==6:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("오른쪽 옆구리 유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==7:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("양쪽 골반유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==8:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("왼쪽 윗다리 유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==9:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("오른쪽 윗다리유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==10:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("왼쪽 아랫다리유사도: ",sim_score,"%")
                sum+=sim_score
            elif m ==11:
                sim_score=score(cam_pose_data[path[i][0]][m],video_pose_data[path[i][1]][m])
                score_i_list.append(sim_score)
                #print("오른쪽 아랫다리 유사도: ",sim_score,"%")
                sum+=sim_score
        score_i_list.append(sum/12)
        score_list.append(np.array(score_i_list))
        score_i_list.clear()
        #print("평균점수 : ",sum/12)
        

real_score_list=[]

for i in range(len(score_list)):
    if len(real_score_list)==0:
        real_score_list.append(np.array(score_list[i]))
        continue
    if real_score_list[-1][0] ==score_list[i][0]:
        if real_score_list[-1][-1]<score_list[i][-1]:
            real_score_list[-1]=score_list[i]
    else:
        real_score_list.append(np.array(score_list[i]))
final_score=0
frame_num=0
line_array=[]
while True:
    
    SUCCESS, me = result_cam.read()
    if  me is None:
         break
    if d1 > d2 :
        me = cv2.flip(me,1) #좌우반전
    imgRGB2=cv2.cvtColor(me,cv2.COLOR_BGR2RGB)
    results2=pose2.process(imgRGB2)
    #print(results.pose_landmarks)
    if results2.pose_landmarks:
        mpDraw2.draw_landmarks(me,results2.pose_landmarks
                              ,mpPose2.POSE_CONNECTIONS)  
        for id, lm in enumerate(results2.pose_landmarks.landmark):
            
            h, w, c = me.shape
            
            lm=results2.pose_landmarks.landmark[id]
            
        
            cx, cy = (lm.x * w),(lm.y * h)
            line_array.append(np.array([cx,cy]))

        cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[12][0]), int(line_array[12][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[12][0]), int(line_array[12][1])), (int(line_array[14][0]), int(line_array[14][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[13][0]), int(line_array[13][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[14][0]), int(line_array[14][1])), (int(line_array[16][0]), int(line_array[16][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[13][0]), int(line_array[13][1])), (int(line_array[15][0]), int(line_array[15][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[12][0]), int(line_array[12][1])), (int(line_array[24][0]), int(line_array[24][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[23][0]), int(line_array[23][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[23][0]), int(line_array[23][1])), (int(line_array[24][0]), int(line_array[24][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[24][0]), int(line_array[24][1])), (int(line_array[26][0]), int(line_array[26][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[23][0]), int(line_array[23][1])), (int(line_array[25][0]), int(line_array[25][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[26][0]), int(line_array[26][1])), (int(line_array[28][0]), int(line_array[28][1])), (0,255,0), 10)
        cv2.line(me, (int(line_array[25][0]), int(line_array[25][1])), (int(line_array[27][0]), int(line_array[27][1])), (0,255,0), 10)


   
        for j in range(len(real_score_list[frame_num])):
            if j==0:
                print(real_score_list[frame_num][j],"번째 프레임")
            elif j==1:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[12][0]), int(line_array[12][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[12][0]), int(line_array[12][1])), (0,0,255), 10)
                print("양쪽어깨 유사도: ",str(real_score_list[frame_num][j]),"%")
                
            
            elif j==2:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[12][0]), int(line_array[12][1])), (int(line_array[14][0]), int(line_array[14][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[12][0]), int(line_array[12][1])), (int(line_array[14][0]), int(line_array[14][1])), (0,0,255), 10)
                print("왼쪽 윗팔 유사도: ",str(real_score_list[frame_num][j]),"%")
                     
            elif j==3:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[13][0]), int(line_array[13][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[13][0]), int(line_array[13][1])), (0,0,255), 10)
                print("오른쪽 윗팔 유사도: ",str(real_score_list[frame_num][j]),"%")
                   
            elif j==4:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[14][0]), int(line_array[14][1])), (int(line_array[16][0]), int(line_array[16][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[14][0]), int(line_array[14][1])), (int(line_array[16][0]), int(line_array[16][1])), (0,0,255), 10)
                print("왼쪽 아래팔 유사도: ",str(real_score_list[frame_num][j]),"%")
                    
            elif j==5:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[13][0]), int(line_array[13][1])), (int(line_array[15][0]), int(line_array[15][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[13][0]), int(line_array[13][1])), (int(line_array[15][0]), int(line_array[15][1])), (0,0,255), 10)
                print("오른쪽 아래팔 유사도: ",str(real_score_list[frame_num][j]),"%")
                
            elif j==6:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[12][0]), int(line_array[12][1])), (int(line_array[24][0]), int(line_array[24][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[12][0]), int(line_array[12][1])), (int(line_array[24][0]), int(line_array[24][1])), (0,0,255), 10)
                print("왼쪽 옆구리 유사도: ",str(real_score_list[frame_num][j]),"%")
                
            elif j==7:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[23][0]), int(line_array[23][1])), (0,107,255), 10)

                elif real_score_list[frame_num][j] < Red_Line:
                     cv2.line(me, (int(line_array[11][0]), int(line_array[11][1])), (int(line_array[23][0]), int(line_array[23][1])), (0,0,255), 10)
                print("오른쪽 옆구리 유사도: ",str(real_score_list[frame_num][j]),"%")
                   
            elif j==8:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[23][0]), int(line_array[23][1])), (int(line_array[24][0]), int(line_array[24][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[23][0]), int(line_array[23][1])), (int(line_array[24][0]), int(line_array[24][1])), (0,0,255), 10)
                print("양쪽 골반 유사도: ",str(real_score_list[frame_num][j]),"%")
                
                      
            elif j==9:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[24][0]), int(line_array[24][1])), (int(line_array[26][0]), int(line_array[26][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[24][0]), int(line_array[24][1])), (int(line_array[26][0]), int(line_array[26][1])), (0,0,255), 10)
                print("왼쪽 윗다리 유사도: ",str(real_score_list[frame_num][j]),"%")
                  
            elif j==10:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[23][0]), int(line_array[23][1])), (int(line_array[25][0]), int(line_array[25][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[23][0]), int(line_array[23][1])), (int(line_array[25][0]), int(line_array[25][1])), (0,0,255), 10)
                print("오른쪽 윗다리 유사도: ",str(real_score_list[frame_num][j]),"%")
                   
            elif j==11:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[26][0]), int(line_array[26][1])), (int(line_array[28][0]), int(line_array[28][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[26][0]), int(line_array[26][1])), (int(line_array[28][0]), int(line_array[28][1])), (0,0,255), 10)
                print("왼쪽 아래다리 유사도: ",str(real_score_list[frame_num][j]),"%")
                   
            elif j==12:
                if real_score_list[frame_num][j] < Orange_Line and real_score_list[frame_num][j] >= Red_Line:
                    cv2.line(me, (int(line_array[25][0]), int(line_array[25][1])), (int(line_array[27][0]), int(line_array[27][1])), (0,107,255), 10)
                elif real_score_list[frame_num][j] < Red_Line:
                    cv2.line(me, (int(line_array[25][0]), int(line_array[25][1])), (int(line_array[27][0]), int(line_array[27][1])), (0,0,255), 10)
                print("오른쪽 아래다리 유사도: ",str(real_score_list[frame_num][j]),"%")
                   
            else:
                print("평균점수 : ",str(real_score_list[frame_num][j]),"%")
                
                final_score+=real_score_list[frame_num][j]
                if real_score_list[frame_num][j] <Red_Line:
                    cv2.waitKey(1000)        

    line_array.clear()
    frame_num+=1
    cv2.imshow("result_cam",me)
    cv2.waitKey(1)
    
    
                    
print(path)   
print("최종점수 : ",final_score/frame_num,"%")                  
print("d1 :" ,d1)
print("d2 :", d2)
print("종료")
cv2.destroyAllWindows()    
