import cv2
import ffmpeg
import pose_tracking
import video_resize_crop



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

cap_path = 'yoga1.mp4'
cap = cv2.VideoCapture(cap_path)          # 따라해야할 영상
top, toe, center = pose_tracking.tracking(cap)   #상단-하단의 길이와 비디오 넓이의 중앙을 대략적으로 계산합니다.
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture(cap_path)
video_resize_crop.resize_crop(cap, top, toe, center)       #영상을 자르고, 리사이징 후 재저장합니다.
cap.release()
cv2.destroyAllWindows()
(
    ffmpeg
    .input('out1.mp4')
    .output('out.mp4', vcodec = 'h264', acodec = 'aac')
    .run()
)
cap_path = 'out.mp4'
cap = cv2.VideoCapture(cap_path)
_,_,_ = pose_tracking.tracking(cap)
cap.release()
cv2.destroyAllWindows()
