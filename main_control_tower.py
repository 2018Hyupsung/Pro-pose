import os
import cv2
import pose_tracking
import pose_tracking_info
import video_less_frame



##########################################################
#                                                        #
#                                                        #
#       Pro-pose(프로, 포즈) : 영상인식기반 자세교정 도구         # 
#       team : 김봉준/장준영/서재원    지도교수 : 홍석주          #
#                                                        #
#                                                        #
##########################################################




#------------------------교수/학습자 영상 제목
instructor = 'yoga1'
student = 'yoga1_1'
#------------------------

#------------------------경로 모음
ins_path = './ins_vid/'
stu_path = './stu_vid/'
csv_path = './csv/'
ins_listdir = os.listdir(ins_path)
stu_listdir = os.listdir(stu_path)
csv_listdir = os.listdir(csv_path)
#------------------------

#------------------------확장자/파일 이름 연결자 모음
mp4 = '.mp4'
csv = '.csv'
less_finished = '_15fps_'
#------------------------

#------------------------기타
already = False
#------------------------






#------------------------- 1.성능 향상을 위해 영상의 프레임을 15frame/s 로 제한합니다.
if (instructor+less_finished+mp4) in ins_listdir :
    already = True
if already == False :
    video_less_frame.less_frame(ins_path+instructor,mp4)
ins_frames = video_less_frame.get_vid_info(ins_path+instructor+less_finished+mp4)


already = False

if (student+less_finished+mp4) in stu_listdir :
    already = True
if already == False :
    video_less_frame.less_frame(stu_path+student,mp4)
stu_frames = video_less_frame.get_vid_info(stu_path+student+less_finished+mp4)

already = False
#-------------------------



#------------------------- 2-1.교수자 영상의 csv데이터를 찾고, 존재하지 않는다면 만들어줍니다.
if (instructor+less_finished+csv) in csv_listdir:
    already = True
if already == False :
    ins_cap = cv2.VideoCapture(ins_path+instructor+less_finished+mp4)
    pose_tracking_info.tracking_info(ins_cap,ins_frames,instructor)     # (추후) 이미 해당영상에 대해 .csv파일이 존재한다면 이 과정을 생략합니다. / 강의 영상에 대한 scv파일을 생성합니다.
    ins_cap.release()
ins_info = pose_tracking.read_ins_info(csv_path, instructor+less_finished,csv)    # csv파일을 불러들입니다.
#-------------------------

already = False

#------------------------- 2-2.학습자 영상의 csv데이터를 찾고, 존재하지 않는다면 만들어줍니다.
if (student+less_finished+csv) in csv_listdir:
    already = True
if already == False :
    stu_cap = cv2.VideoCapture(stu_path+student+less_finished+mp4)
    pose_tracking_info.tracking_info(stu_cap,stu_frames,student)     # (추후) 이미 해당영상에 대해 .csv파일이 존재한다면 이 과정을 생략합니다. / 강의 영상에 대한 scv파일을 생성합니다.
    stu_cap.release()
stu_info = pose_tracking.read_ins_info(csv_path, student+less_finished,csv)    # csv파일을 불러들입니다.
#-------------------------



#------------------------- 3.교수자의 데이터와 학습자의 영상을 비교분석합니다.
stu_cap = cv2.VideoCapture(stu_path+student+less_finished+mp4)
pose_tracking.tracking(ins_info, stu_info, stu_cap)
stu_cap.release()
#-------------------------
