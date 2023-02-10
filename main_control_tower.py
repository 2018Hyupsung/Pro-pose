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
###########################################################


# 웹캠 초기화. 오류가 발생한다면 -1, 0, 1 중 하나로 초기화 시켜봅니다.
# cap = cv2.VideoCapture(1)
instructor = 'yoga1'
ins_path = './ins_vid/'
student = 'yoga1_1'
stu_path = './stu_vid/'
csv_path = './csv/'
vid = '.mp4'
info = '.csv'
ins_out_name, frames = video_less_frame.less_frame(ins_path+instructor,vid)    # 성능향상을 위해 영상의 프레임 수를 제한합니다.
stu_out_name, frames = video_less_frame.less_frame(stu_path+student,vid)

ins_cap = cv2.VideoCapture(ins_out_name)          # 따라해야할 영상    
stu_cap = cv2.VideoCapture(stu_out_name)

pose_tracking_info.tracking_info(ins_cap, frames, instructor)                   # (추후) 이미 해당영상에 대해 .csv파일이 존재한다면 이 과정을 생략합니다. / 강의 영상에 대한 scv파일을 생성합니다.
ins_info = pose_tracking.read_ins_info(csv_path, instructor,info)               # csv파일을 불러들입니다.
pose_tracking.tracking(ins_info, stu_cap)
ins_cap.release()
stu_cap.release()

