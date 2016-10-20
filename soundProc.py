import wave
import contextlib
import os
import glob
import subprocess
import shutil

def getAudioDuration(filename):
    with contextlib.closing(wave.open(filename,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return(duration)

if __name__ == '__main__':
    FILE_PATH_FFMPEG_EXE = "ffmpeg-3.1.4-win64-static/bin"
    FILE_PATH_AUDIO_TO_CONVERT = "vine/trainingAudio"
    TARGET_DURATION = 6.0
    
    os.chdir(FILE_PATH_AUDIO_TO_CONVERT)

    
    newpath = 'converted'
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    #s = 0
    for audioFile in glob.glob("*.wav"):
        audioTime = getAudioDuration(audioFile)
        os.chdir("../..")
        os.chdir(FILE_PATH_FFMPEG_EXE)

        exe = "./ffmpeg"
        audioFileArg = "../../" + FILE_PATH_AUDIO_TO_CONVERT + "/" + audioFile
        rate = audioTime / TARGET_DURATION
        
        if (rate < 0.5):
            rate = 0.5
        rateArg = "atempo=" + str(rate)
        outputFileArg = "../../" + FILE_PATH_AUDIO_TO_CONVERT + "/" + newpath + "/" + audioFile
        
        args = [exe, "-i", audioFileArg, "-filter:a", rateArg, "-vn", outputFileArg]
        #print (args)
        
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(args, stdout=FNULL, stderr=FNULL, shell =False)
        #s += 1
        #print ("Completed: " + str(s))
        
        os.chdir("../..")
        os.chdir(FILE_PATH_AUDIO_TO_CONVERT)
        # ./ffmpeg -i 1005958271451971584.wav -filter:a "atempo=2.0" -vn output.wav
    os.chdir("../..")
        
