import subprocess

subprocess.call("python preprocessVideos.py")

from preprocessVideos import convertWav
print ("convert...")
convertWav()
