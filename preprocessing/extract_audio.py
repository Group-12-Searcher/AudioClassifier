__author__ = "xiangwang1223@gmail.com"
# The simple implementation of obtaining the audio clip of a original video.

import glob
import os
import moviepy.editor as mp


def getAudioClip(video_reading_path, audio_storing_path):
    clip = mp.VideoFileClip(video_reading_path)
    clip.audio.write_audiofile(audio_storing_path)


if __name__ == '__main__':
    os.chdir("../")
    start = False
    for videoPath in glob.glob("vine/training/*.mp4"):
        if (videoPath.split("\\")[-1] == "1000046931730481152.mp4"):
            start = True
        if (start == False):
            continue
        audioPath = videoPath.split(".")[0] + ".wav"
        getAudioClip(video_reading_path=videoPath, audio_storing_path=audioPath)
        
    '''# 1. Set the access path to the original file.
    video_reading_path = "../data/video/1.mp4"

    # 2. Set the path to store the extracted audio clip.
    audio_storing_path = "../data/audio/1.wav"

    # 3. Fetch and store the corresponding audio clip.
    getAudioClip(video_reading_path=video_reading_path, audio_storing_path=audio_storing_path)
'''
