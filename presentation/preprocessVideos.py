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

if __name__ == '__main__':
    print ("Generating frames from videos...")
    preprocessFrames()
    print ("Generating wav from videos...")
    preprocessWav()
