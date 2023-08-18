import os
from fnmatch import fnmatch
import subprocess

root = '/home/sail/Downloads/Team_A_data_logs'

vids = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if 'compressed.mp4' not in name and fnmatch(name, '*.mp4') or fnmatch(name, '*.MP4'):
            f = os.path.join(path, name)
            vids.append(f)

cmd = ''
for v in vids:
    v=v.replace(' ', '\ ')
    cmd+= ';'
    # cmd += ' '.join([
    #     'ffmpeg', 
    #     '-hwaccel',
    #     'cuda',
    #     '-i', v, 
    #     '-vcodec', 
    #     # 'libx265', 
    #     'h264_nvenc',
    #     # '-crf', 
    #     # '28', 
    #     v[:-4]+'compressed.mp4;', 
    #     'rm', 
    #     v])
    cmd += ' '.join([
        'ffmpeg', 
        '-hwaccel',
        'cuda',
        '-hwaccel_output_format',
        'cuda',
        '-i', v, 
        '-c:v',
        'h264_nvenc',
        v[:-4]+'compressed.mp4;', 
        'rm', 
        v])

print(cmd)