import ffmpeg
from ffprobe import FFProbe
from math import trunc





def get_vid_info(vid) :
    
    metadata=FFProbe(vid)

    for stream in metadata.streams:
        if stream.is_video():
            frames = stream.frames()
            durations = trunc(stream.duration_seconds())
            fps = round(frames / durations)
            print('총 프레임 : {} frames.'.format(frames))
            print('총 재생시간 : {} seconds.'.format(durations))
            print('fps : {} frames/s.'.format(fps))

    return frames





def less_frame(vid, mp4) :

    if(mp4 == '.mov') :
        mp4 = '.mp4'
        out_name = vid + mp4
        (
        ffmpeg
        .input(vid+'.mov')
        .filter('fps', fps=15, round='up')
        .hflip()
        .output(out_name, vcodec = 'h264', acodec = 'aac')
        .run()
        )

    out_name = vid + '_15fps_' + mp4
    print(vid+mp4)
    metadata=FFProbe(vid+mp4)
    for stream in metadata.streams:
        if stream.is_video():
            frames = stream.frames()
            durations = trunc(stream.duration_seconds())
            fps = round(frames / durations)

    #15프레임으로 드롭
    durations = trunc(durations * 15 / fps)

    (
    ffmpeg
    .input(vid+mp4)
    .filter('fps', fps=15, round='up')
    #.filter('scale', 640, -2)
    #.setpts('1.0/{}*PTS'.format(fps-18))       #줄어든 프레임 수 만큼 영상의 총 재생 시간을 감소시킵니다.
    .output(out_name, vcodec = 'h264', acodec = 'aac')
    #.output(out_name, vcodec = 'h264', acodec = 'aac', t = durations)

    .run()
    )
    

    return out_name