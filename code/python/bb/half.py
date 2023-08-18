import os
from fnmatch import fnmatch
import subprocess

vis_root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/visible'
invis_root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/invisible'
vis_dest = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/frame_vis/'
invis_dest = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/frame_invis/'

vids = []

for path, subdirs, files in os.walk(vis_root):
    for name in files:
        if '.mp4' not in name.lower():
            continue
        f = os.path.join(path, name)
        outname = vis_dest+name[:-4] + '.png'
        cmd = 'input='+f+';'
        cmd += ' '.join([
            'ffmpeg',
            '-ss',
            "\"$(bc -l <<< \"$(ffprobe -loglevel error -of csv=p=0 -show_entries format=duration \"$input\")*0.5\")\"",
            '-i',
            "\"$input\"",
            "-frames:v",
            "1",
            outname
        ])
        cmd+=';'

        print(cmd)
        
for path, subdirs, files in os.walk(invis_root):
    for name in files:
        if '.mp4' not in name.lower():
            continue
        f = os.path.join(path, name)
        outname = invis_dest+name[:-4] + '.png'
        cmd = 'input='+f+';'
        cmd += ' '.join([
            'ffmpeg',
            '-ss',
            "\"$(bc -l <<< \"$(ffprobe -loglevel error -of csv=p=0 -show_entries format=duration \"$input\")*0.5\")\"",
            '-i',
            "\"$input\"",
            "-frames:v",
            "1",
            outname
        ])
        cmd+=';'

        print(cmd)
