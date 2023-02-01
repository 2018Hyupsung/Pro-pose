import ffmpeg

def less_frame(cap) :
    (
    ffmpeg
    .input(cap)
    .filter('fps', fps=18, round='down')
    .output('ed_frame.mp4', vcodec = 'h264', acodec = 'aac')
    .run()
    )