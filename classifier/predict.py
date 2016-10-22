import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.metrics import classification_report
import os
from sklearn.externals import joblib

# Train your own classifier.
# Here is the simple implementation of SVM classification.
def predictSVM(name, isMain = False):
    os.chdir("../")
    validateFolder = "presentation"
    
    print("Loading test data...")
    mfccTestData = np.loadtxt(validateFolder+"/mfccMean.csv", delimiter=",")
    spectTestData = np.loadtxt(validateFolder+"/spectMean.csv", delimiter=",")
    zeroTestData = np.loadtxt(validateFolder+"/zero.csv", delimiter=",")
    energyTestData = np.loadtxt(validateFolder+"/energy.csv", delimiter=",")
    #X_test = np.concatenate((mfccTestData, spectTestData, zeroTestData, energyTestData), axis=1)
    X_test = np.concatenate((spectTestData, mfccTestData), axis=1)

    numFiles = X_test.shape[0]
    numClasses = 30
    classScores = [[0]*numClasses for i in xrange(numFiles)]
    
    for classNum in range(1, numClasses+1):
        print("Loading classifier...")
        model = joblib.load('classifier/savedmodel' + str(classNum) + '.pkl')

        print("Predicting classes for test data...")
        Y_predicted = np.asarray(model.predict(X_test))
        Y_scores = model.decision_function(X_test)
            
        print('SVM Predict Done.')

        print("Saving result...\n")
        numFiles = Y_predicted.shape[0]
        for i in range(numFiles):
            classScores[i][classNum-1] = Y_scores[i]
        
        '''print("Saving result...")
        print Y_files.shape, Y_predicted.shape, Y_truth.shape
        result = np.transpose(np.vstack((Y_files, Y_predicted, Y_truth)))
        np.savetxt("prediction.csv", result, delimiter=",")
        np.savetxt("prediction_scores.csv", Y_scores, delimiter=",")'''
        
    print("Done!")

    topClassesCollection = []
    for i in range(numFiles):
        topClasses = []
        for j in range(5):
            topClasses.append(classScores[i].index(max(classScores[i])) + 1)
            classScores[i][classScores[i].index(max(classScores[i]))] = -999
        topClassesCollection.append(topClasses)

    if (isMain):
        print topClassesCollection
    else:
        index = -1
        with open('presentation/nameToIndex.txt', 'r') as mapFile:
            for line in mapFile:
                parts = line.split(" ")
                extractedName = parts[0]
                extractedIndex = parts[1]
                if (extractedName == name):
                    index = int(extractedIndex)
                    return topClassesCollection[index]        

if __name__ == '__main__':
    predictSVM("", True)
    #print(predictSVM("1.wav"))
