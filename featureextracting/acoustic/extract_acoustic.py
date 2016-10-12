# __author__ = "xiangwang1223@gmail.com"
# The simple implementation of extracting multiple types of traditional acoustic features,
# consisting of Mel-Frequency Cepstral Coefficient (MFCC), Zero-Crossing Rate, Melspectrogram,
#  and Root-Mean-Square features.

# Input: an original audio clip.
# Output: multiple types of acoustic features.
#   Please note that, 1. you need to select suitable and reasonable feature vector(s) to represent the video.
#                     2. if you select mfcc features, you need to decide how to change the feature matrix to vector.

# More details: http://librosa.github.io/librosa/tutorial.html#more-examples.

from __future__ import print_function
import moviepy.editor as mp
import librosa
import numpy as np
import glob
import os
import csv

def getAcousticFeatures(audio_reading_path, getMfcc, getSpect, getZero, getEnergy):
    # 1. Load the audio clip;
    y, sr = librosa.load(audio_reading_path)

    # 2. Separate harmonics and percussives into two waveforms.
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # 3. Beat track on the percussive signal.
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    feature_mfcc = None
    feature_spect = None
    feature_zero = None
    feature_energy = None
    dataLength = None

    if getMfcc:
        # 4. Compute MFCC features from the raw signal.
        feature_mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)

    if getSpect:
        # 5. Compute Melspectrogram features from the raw signal.
        feature_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=80000)

    if getZero:
        # 6. Compute Zero-Crossing features from the raw signal.
        feature_zero = librosa.feature.zero_crossing_rate(y=y)
        
    if getEnergy:
        # 7. Compute Root-Mean-Square (RMS) Energy for each frame.
        feature_energy = librosa.feature.rmse(y=y)
        
    if getEnergy:
        dataLength = int(np.shape(feature_energy)[1])
    elif getMfcc:
        dataLength = int(np.shape(feature_mfcc)[1])
    elif getSpect:
        dataLength = int(np.shape(feature_spect)[1])
    elif getZero:
        dataLength = int(np.shape(feature_zero)[1])
                     
    return feature_mfcc, feature_spect, feature_zero, feature_energy, dataLength


if __name__ == '__main__':
    os.chdir("../../")
    
    ### YOU CAN CHANGE THIS TO YOUR OWN FOLDERS' PATHS ###
    audioFolderpath = "vine/trainingAudio/"
    storageFolderpath = "feature/acoustic/"

    ### CONSTANT VARIABLES ###
    FIXED_DATALENGTH = 300   # The length of a data after doing zero-padding
    MFCC_NUMVECTORS = 13   # The default number of vectors of a MFCC feature
    SPECT_NUMVECTORS = 82    # The default number of vectors of a spect feature (we ignore the last 46 vectors, which are all zeros)
    MFCCMEAN_NUMGROUP_PERVECTOR = 20   # We split each MFCC vector into 20 groups, and then get the mean of each group
    SPECTMEAN_NUMGROUP_PERVECTOR = 3   # We split each spect vector into 3 groups, and then get the mean of each group
    MFCCMEAN_DATALENGTH = MFCC_NUMVECTORS * MFCCMEAN_NUMGROUP_PERVECTOR  # 13*20=260
    SPECTMEAN_DATALENGTH = SPECT_NUMVECTORS * SPECTMEAN_NUMGROUP_PERVECTOR   # 82*3=246

    ### Set 'True' if you want to extract the feature, 'False' otherwise ###
    getMfccMean = True
    getSpectMean = True
    getZero = True
    getEnergy = True
    
    print("Extracting these acoustic features:")
    if getMfccMean: print("\tMFCC (mean)")
    if getSpectMean: print("\tMelspectrogram (mean)")
    if getZero: print("\tZero-Crossing Rate")
    if getEnergy: print("\tRMS Energy")
    print("-----------------------------------")

    numFiles = 0   # Set the number of audio files used (Set to 0 to use ALL available files)
    if numFiles <= 0:
        print("Gathering data from ALL audio files...")
    else:
        print("Gathering data from", numFiles, "audio files...")
        
    mfccData = []
    spectData = []
    zeroData = []
    energyData = []
    mfccMeanData = []
    spectMeanData = []
    dataLengths = []   # Stores the data lengths of all audio files
    s = 0   # Track the number of files that have been processed
    
    os.chdir(audioFolderpath)
    for audioFile in glob.glob("*.wav"):
        audioFile = audioFile.replace("\\", "/")
        feature_mfcc, feature_spect, feature_zero, feature_energy, dataLength = getAcousticFeatures(
            audioFile, getMfccMean, getSpectMean, getZero, getEnergy)

        if getEnergy:
            ## Read values from feature into list ##
            data = []
            for x in np.nditer(feature_energy):
                data.append(x)
                
            ## Do zero-padding ##
            for i in range(FIXED_DATALENGTH - dataLength):
                data.append(0)   
                
            energyData.append(data)  # Add data to result

        if getZero:
            ## Read values from feature into list ##
            data = []
            for x in np.nditer(feature_zero):
                data.append(x)

            ## Do zero-padding ##
            for i in range(FIXED_DATALENGTH - dataLength):
                data.append(0)

            zeroData.append(data)  # Add data to result

        if getMfccMean:
            ## Read all values from all the vectors of the feature into a single list ##
            concData = []
            for x in np.nditer(feature_mfcc):
                concData.append(x.item(0))
            
            meanData = []
            groupSize = dataLength/MFCCMEAN_NUMGROUP_PERVECTOR
            k = 0  # The start index of the current group
            g = MFCCMEAN_NUMGROUP_PERVECTOR  # The number of groups left for the current vector
            r = dataLength % MFCCMEAN_NUMGROUP_PERVECTOR  # The remainder after dividing by the number of groups per vector
            
            for i in range(MFCCMEAN_DATALENGTH):
                dataGroup = []

                if g == r:
                    groupSize += 1   # Increases the group size by 1 for the remaining groups of the current vector
                if g == 0:
                    g = MFCCMEAN_NUMGROUP_PERVECTOR   # reset group count for new vector
                    groupSize -= 1   # reset group size for new vector

                ## Get the mean of the current group ##
                for j in range(k, k + groupSize):
                    dataGroup.append(concData[j])
                meanData.append(sum(dataGroup)/groupSize)

                ## Move on to the next group ##
                k += groupSize  
                g -= 1

            ## Do zero-padding ##
            for i in range(FIXED_DATALENGTH - MFCCMEAN_DATALENGTH):
                meanData.append(0)
                
            mfccMeanData.append(meanData)  # Add data to result

        if getSpectMean:
            ## Read all values from all the vectors of the feature into a single list ##
            concData = []
            for x in np.nditer(feature_spect):
                concData.append(x.item(0))
                
            meanData = []
            groupSize = dataLength/SPECTMEAN_NUMGROUP_PERVECTOR
            k = 0  # The start index of the current group
            g = SPECTMEAN_NUMGROUP_PERVECTOR  # The number of groups left for the current vector
            r = dataLength % SPECTMEAN_NUMGROUP_PERVECTOR  # The remainder after dividing by the number of groups per vector
            
            for i in range(SPECTMEAN_DATALENGTH):
                dataGroup = []
                
                if g == r:
                    groupSize += 1   # Increases the group size by 1 for the remaining groups of the current vector
                if g == 0:
                    g = SPECTMEAN_NUMGROUP_PERVECTOR   # reset group count for new vector
                    groupSize -= 1   # reset group size for new vector

                ## Get the mean of the current group ##
                for j in range(k, k + groupSize):
                    dataGroup.append(concData[j])
                meanData.append(sum(dataGroup) / groupSize)

                ## Move on to the next group ##
                k += groupSize
                g -= 1

            ## Do zero-padding ##  
            for i in range(FIXED_DATALENGTH - SPECTMEAN_DATALENGTH):
                meanData.append(0)
                
            spectMeanData.append(meanData)  # Add data to result

        s += 1
        audioName = audioFile.split("/")[-1].split(".")[0]
        print("{}) {}.wav Done. Data Length: {}".format(s, audioName, dataLength))
        dataLengths.append(dataLength)
        
        if s == numFiles: break
        
    print("Max Data Length:", max(dataLengths))
    os.chdir("../../")

    ### WRITE RESULTS TO CSV FILES ###
    os.chdir(storageFolderpath)
    print("Writing data to file...")
    if getEnergy:
        energyData = np.array(energyData)
        np.savetxt("energy.csv", energyData, delimiter=",")
    if getZero:
        zeroData = np.array(zeroData)
        np.savetxt("zero.csv", zeroData, delimiter=",")
    if getMfccMean:
        mfccMeanData = np.array(mfccMeanData)
        np.savetxt("mfccMean.csv", mfccMeanData, delimiter=",")
    if getSpectMean:
        spectMeanData = np.array(spectMeanData)
        np.savetxt("spectMean.csv", spectMeanData, delimiter=",")
    print("Done!")
        
