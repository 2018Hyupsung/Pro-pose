import cv2
import ffmpeg
import pose_tracking_learner
import pose_tracking_info
import video_less_frame



##########################################################
#                                                         
#       Pro-pose(프로, 포즈) : 영상인식기반 자세교정 시스템         
#       팀장/팀원 : 김봉준/장준영/서재원    지도교수 : 홍석주        
#       
#
#
###########################################################


# 웹캠 초기화. 오류가 발생한다면 -1, 0, 1 중 하나로 초기화 시켜봅니다.
# cap = cv2.VideoCapture(1)

demonstrator = 'yoga1'
vid = '.mp4'
out_name = video_less_frame.less_frame(demonstrator,vid)
print(out_name)
cap = cv2.VideoCapture(out_name)          # 따라해야할 영상
#pose_tracking_info.tracking_info(cap)
#cap.release()
#cv2.destroyAllWindows()

#video_less_frame.less_frame(ori_path1)             # 성능향상을 위해 영상의 프레임 수를 제한합니다.
#ori_path1 = 'ed_frame.mp4'
#cap = cv2.VideoCapture(ori_path1)
pose_tracking_learner.tracking(cap)
cap.release()
#cv2.destroyAllWindows()
