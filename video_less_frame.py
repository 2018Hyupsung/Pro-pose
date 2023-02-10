import ffmpeg
from ffprobe import FFProbe
from math import trunc

def less_frame(vid,mp4) :
    out_name = vid + '_15fps_' + mp4
    frames = 0
    durations = 0
    fps = 0

    metadata=FFProbe(vid+mp4)

    for stream in metadata.streams:
        if stream.is_video():
            frames = stream.frames()
            durations = trunc(stream.duration_seconds())
            fps = round(frames / durations)
            print('{} frames.'.format(frames))
            print('{} seconds.'.format(durations))
            print('{} frames/s.'.format(fps))
    
    #15프레임으로 드롭
    durations = trunc(durations * 15 / fps)
    print('new vid''s durations = {}'.format(durations))

    (
    ffmpeg
    .input(vid+mp4)
    .filter('fps', fps=15, round='up')
    #.setpts('1.0/{}*PTS'.format(fps-18))       #줄어든 프레임 수 만큼 영상의 총 재생 시간을 감소시킵니다.
    .output(out_name, vcodec = 'h264', acodec = 'aac')
    #.output(out_name, vcodec = 'h264', acodec = 'aac', t = durations)
    .run()
    )
    
    metadata=FFProbe(out_name)

    for stream in metadata.streams:
        if stream.is_video():
            frames = stream.frames()
            durations = trunc(stream.duration_seconds())
            fps = round(frames / durations)
            print('{} frames.'.format(frames))
            print('{} seconds.'.format(durations))
            print('{} frames/s.'.format(fps))


    return out_name, frames