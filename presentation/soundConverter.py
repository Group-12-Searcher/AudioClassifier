import os
import shutil
import sys
import glob
import subprocess
os.chdir("../")
sys.path.append(os.getcwd()) # needed to import code below.
import soundProc

FILE_PATH_FFMPEG_EXE = "ffmpeg-3.1.4-win64-static/bin"
FILE_PATH_AUDIO_TO_CONVERT = "presentation"
TARGET_DURATION = 6.0
    
os.chdir(FILE_PATH_AUDIO_TO_CONVERT)   
newpath = 'converted'
if os.path.exists(newpath):
    shutil.rmtree(newpath)
if not os.path.exists(newpath):
    os.makedirs(newpath)

for audioFile in glob.glob("*.wav"):
    print (audioFile)
    audioTime = soundProc.getAudioDuration(audioFile)
    os.chdir("../")
    os.chdir(FILE_PATH_FFMPEG_EXE)

    exe = "./ffmpeg"
    audioFileArg = "../../" + FILE_PATH_AUDIO_TO_CONVERT + "/" + audioFile
    rate = audioTime / TARGET_DURATION
        
    if (rate < 0.5):
        rate = 0.5
    rateArg = "atempo=" + str(rate)
    outputFileArg = "../../" + FILE_PATH_AUDIO_TO_CONVERT + "/" + newpath + "/" + audioFile
        
    args = [exe, "-i", audioFileArg, "-filter:a", rateArg, "-vn", outputFileArg]
        
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(args, stdout=FNULL, stderr=FNULL, shell =False)
        
    os.chdir("../..")
    os.chdir(FILE_PATH_AUDIO_TO_CONVERT)
    # ./ffmpeg -i input.wav -filter:a "atempo=2.0" -vn output.wav
os.chdir("../")
