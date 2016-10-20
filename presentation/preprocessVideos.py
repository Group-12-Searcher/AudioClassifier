import cv2
import os
import sys
import glob
import numpy as np
os.chdir("../")
sys.path.append(os.getcwd()) # needed to import code below.
from preprocessing.extract_frame import getKeyFrames
from preprocessing.extract_audio import getAudioClip
import soundProc
import shutil
import subprocess

### WARNING : DO NOT RUN THIS CODE DIRECTLY, RUN THE mainpreprocess.py instead. ###

def preprocessFrames():
    os.chdir("presentation")

    for videoFile in glob.glob("*.mp4"):
        # Open the video clip.
        vidclip = cv2.VideoCapture(videoFile)
        # Get and store the resulting frames via the specific path.
        storePath = "frame/" + videoFile.split(".")[0] + "-"
        keyframes = getKeyFrames(vidcap=vidclip, store_frame_path=storePath )
        # Close the video clip.
        vidclip.release()
    os.chdir("../")

def preprocessWav():
    os.chdir("presentation")

    for videoFile in glob.glob("*.mp4"):
        resultAudioPath = videoFile.split(".")[0] + ".wav"
        # Fetch and store the corresponding audio clip.
        try:
            getAudioClip(video_reading_path=videoFile, audio_storing_path=resultAudioPath)
        except IOError: # Music is shorter than video duration error  
            print ("\n")
            print ("This is an error where music file stops before video file ends, ")
            print ("please do NOT panic!!! All is fine :) Message as follows : ")
            print (sys.exc_info()[1])
        
    os.chdir("../")

def convertWav():
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

if __name__ == '__main__':
    print ("Generating frames from videos...")
    preprocessFrames()
    print ("Generating wav from videos...")
    preprocessWav()
    #print ("Converting wav to uniform wav versions...")
    #convertWav()
    #print (os.getcwd())
