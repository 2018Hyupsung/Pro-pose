import cv2


def resize_crop(cap, top, toe, image_weight) :

    top = round(top - (top*0.30))                    #화면 잘림 방지를 위해 영상 crop시 여유 비율을 30%로 계산합니다.
    toptotoe = round((toe - top)/2)
    image_weight = round(image_weight/2)
    crop_frame = None
    print("최고 : ", top, ", 최저 : ", toe)

    # 영상저장용
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('out1.mp4', fourcc, 29.9, (640, 640))


    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        
        crop_frame = frame[top:toe, (image_weight-toptotoe):(image_weight+toptotoe)]
        if (crop_frame.shape[0] < 640) :
            resize_frame = cv2.resize(crop_frame, (640,640),
            interpolation=cv2.INTER_LINEAR)
        elif (crop_frame.shape[0] > 640) :
            resize_frame = cv2.resize(crop_frame, (640,640),
            interpolation=cv2.INTER_AREA)

        # 영상저장
        out.write(resize_frame)

        # 프레임 출력
        cv2.imshow("resize_frame", resize_frame)
        
        # 'q' 를 입력하면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()