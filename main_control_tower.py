import os
import cv2
import read_csv
import pose_tracking_info
import pose_tracking_drawing
import pose_tracking_dtw
import video_less_frame
import merge_csv
import multiprocessing

##########################################################
#                                                        #
#                                                        #
#       Pro-pose(프로, 포즈) : 영상인식기반 자세교정 도구         # 
#       team : 김봉준/장준영/서재원    지도교수 : 홍석주          #
#                                                        #
#                                                        #
##########################################################


if __name__ == '__main__' :


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
    mov = '.mov'
    csv = '.csv'
    less_finished = '_15fps_'
    land_finished = '_land'
    #------------------------

    #------------------------프레임
    ins_frames = 0
    stu_frames = 0
    #------------------------

    #------------------------멀티프로세싱
    is_stu = False
    ins_thread = [int(0)]
    stu_thread = [int(0)]
    ins_infoall = []
    stu_infoall = []
    #------------------------

    #------------------------csv
    ins_csv = []
    stu_csv = []
    #------------------------

    #------------------------dtw - 학생 동작 시작 프레임 찾기
    stu_dtw_frames = []
    #------------------------

    #------------------------기타
    already = False
    #------------------------






    #------------------------- 1.성능 향상을 위해 영상의 프레임을 15frame/s 로 제한합니다.
    if (instructor+less_finished+mp4) in ins_listdir :
        already = True
    if already == False :
        video_less_frame.less_frame(ins_path+instructor,mp4)
    ins_frames = video_less_frame.get_vid_info(ins_path+instructor+less_finished+mp4)-1
    ins_frames = int(ins_frames)
    #cv2의 프레임은 0~끝프레임-1

    already = False

    if (student+less_finished+mp4) in stu_listdir :
        already = True
    if already == False :
        video_less_frame.less_frame(stu_path+student,mp4)
    stu_frames = video_less_frame.get_vid_info(stu_path+student+less_finished+mp4)-1
    stu_frames = int(stu_frames)

    already = False
    #-------------------------




    #-------------------------
    temp = ins_frames
    quot = temp // 7
    for i in range (7) :
        if(ins_thread[i]+quot >= ins_frames) :
            break
        ins_thread.append(ins_thread[i]+quot)
    ins_thread.append(ins_frames)

    temp = stu_frames
    quot = temp // 7
    for i in range (7) :
        if(stu_thread[i]+quot >= stu_frames) :
            break
        stu_thread.append(stu_thread[i]+quot)
    stu_thread.append(stu_frames)
    #-------------------------




    #------------------------- 2.영상의 csv데이터를 찾고, 존재하지 않는다면 만들어줍니다.
    if (instructor+less_finished+csv) in csv_listdir:
        already = True
    if already == False :
        ins_cap = ins_path+instructor+less_finished+mp4

        for idx, val in enumerate (ins_thread) :
            if val is ins_thread[-2] :
                ins_infoall.append((ins_cap, val, ins_frames, is_stu))
                break
            else :
                ins_infoall.append((ins_cap, val, ins_thread[idx+1]-1, is_stu))

        pool = multiprocessing.Pool(processes=12)
        pool.starmap(pose_tracking_info.tracking_info,ins_infoall)
        pool.close()
        pool.join()
        
        merge_csv.merge()

        pool = multiprocessing.Pool(processes=7)
        pool.starmap(pose_tracking_info.tracking_info,ins_infoall)
        pool.close()
        pool.join()
    
        merge_csv.merge()

    already = False

    if (student+less_finished+csv) in csv_listdir:
        already = True
    if already == False :
        is_stu = True
        stu_cap = stu_path+student+less_finished+mp4

        for idx, val in enumerate (stu_thread) :
            if val is stu_thread[-2] :
                stu_infoall.append((stu_cap, val, stu_frames, is_stu))
                break
            else :
                stu_infoall.append((stu_cap, val, stu_thread[idx+1]-1, is_stu))
   
        pool = multiprocessing.Pool(processes=12)
        pool.starmap(pose_tracking_info.tracking_info,stu_infoall)
        pool.close()
        pool.join()

        merge_csv.merge()
        merge_csv.merge_land()
    
    stu_info = read_csv.read_csv(csv_path, student+less_finished,csv)    # csv파일을 불러들입니다.
    ins_info = read_csv.read_csv(csv_path, instructor+less_finished,csv)    # csv파일을 불러들입니다.
    land_info = read_csv.read_csv(csv_path, student+land_finished,csv)

    #pose_tracking_drawing.pose_drawing(stu_path+student+less_finished+mp4, land_info, 0, stu_frames)

    #stu_frame_list = pose_tracking_dtw.tracking_dtw(ins_info, stu_info, stu_path+student+less_finished+mp4, ins_frames)
    #print(stu_frame_list)

    #------------------------- 3.교수자의 데이터와 학습자의 영상을 비교분석합니다.
    stu_cap = cv2.VideoCapture(stu_path+student+less_finished+mp4)
    stu_frame_list = pose_tracking_dtw.tracking_dtw(ins_info, stu_info, stu_cap, ins_frames)
    print(stu_frame_list[0])
    for i in range(ins_frames):
        if(i in stu_frame_list[0]):
            continue
        elif(os.path.exists('./images/frame%d.jpg' % i)):
            print("i")
            os.remove('./images/frame%d.jpg' % i)
    count = 0

    for j in stu_frame_list[0]:
        img = cv2.imread('./images/frame%d.jpg' % j, cv2.IMREAD_UNCHANGED)
        if (j == 0):
            continue
        for i in range(12):
            scores_ = "score " + str(i) + " : " + "{:.2f}".format(stu_frame_list[1][count][i])
            cv2.putText(img, scores_, (50,50 + (i * 20)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        count += 1
        cv2.imwrite("./images/frame%d.jpg" % j,img)
    stu_cap.release()
    #-------------------------
