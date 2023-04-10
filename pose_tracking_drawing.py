import cv2
from media_pipe_module import mediapipe_pose



def pose_drawing(path, land_info, start, end) :

    # 랜드마크 간 연결 리스트
    connects_list = [[0, 1], [3, 1], [2, 0], [5, 3], [4, 2], 
                     [7, 1], [6, 0], [6, 7], [9, 7], [8, 6], 
                     [11, 9], [10, 8]]



    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_now = start


    with mediapipe_pose.Pose(
        min_detection_confidence = 0.5,     # 사람이라고 간주할 수 있는 모델의 최소 신뢰값입니다.
        min_tracking_confidence = 0.5,      # 포즈의 랜드마크를 추적할 수 있는 모델의 최소 신뢰값입니다.
        model_complexity = 1,               # 포즈 랜드마크 모델의 복합성입니다.
        ) as pose :
       
       
        while ((cap.isOpened()) and (frame_now < end-start+1)) :
            success, image = cap.read()

            if not success :
                break
           

            image.flags.writeable = False
            result = pose.process(image)


            # for idx, land in enumerate(connects_list) :
            #     if all(land_info[frame_now][2*(land[i])+j] is not None for i in range(2) for j in range(2)) :
            #         color = (255, 0, 0)
            #         cv2.line(
            #             image,
            #             (land_info[frame_now][2*(land[0])], land_info[frame_now][2*(land[0])+1]),
            #             (land_info[frame_now][2*(land[1])], land_info[frame_now][2*(land[1])+1]),
            #             color = color,
            #             thickness = 7
            #         )
            #     else:
            #         pass

            for idx, land in enumerate(connects_list):
                if any(land_info[frame_now][2*(land[i])+j] is None for i in range(2) for j in range(2)):
                    continue
                    
                color = (255, 0, 0)
                coord1 = (int(round(land_info[frame_now][2*(land[0])])), int(round(land_info[frame_now][2*(land[0])+1])))
                coord2 = (int(round(land_info[frame_now][2*(land[1])])), int(round(land_info[frame_now][2*(land[1])+1])))

                cv2.line(
                    image,
                    coord1,
                    coord2,
                    color=color,
                    thickness=7
                )

            frame_now += 1

            cv2.imshow('Pro-pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()

