import os
import glob

print ("Removing csv files...")
for csvFile in glob.glob("*.csv"):
    os.remove(csvFile)
print ("Removing wav files (original)...")
for wavFile in glob.glob("*.wav"):
    os.remove(wavFile)
print ("Removing jpg frames...")
for frames in glob.glob("frame/*.jpg"):
    os.remove(frames)
print ("Removing wav files (converted)...")
for wavFile in glob.glob("converted/*.wav"):
    os.remove(wavFile)
print ("Remove text file for mapping")
os.remove("nameToIndex.txt")
print ("All cleanup done... Ready for presentation!")
