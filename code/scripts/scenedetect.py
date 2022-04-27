import glob
import os

rootdir = "/atm/"
files = glob.glob(rootdir + "/**/*.mkv", recursive=True)
files_webm = glob.glob(rootdir + "/**/*.webm", recursive=True)
files_mp4 = glob.glob(rootdir + "/**/*.mp4", recursive=True)
files.extend(files_webm)
files.extend(files_mp4)


for f in files:
    os.system(
        f'scenedetect --input "{f}" detect-content -t 1 list-scenes split-video --copy'
    )
    os.remove(f)
