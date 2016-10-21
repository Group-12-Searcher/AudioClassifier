import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.metrics import classification_report
import os

# Trian your own classifier.
# Here is the simple implementation of SVM classification.
def mySVM():
    os.chdir("../")
    trainFolder = "train_converted"
    validateFolder = "validate_converted"

    print("Loading training data...")
    mfccTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/mfccMean.csv", delimiter=",")
    spectTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/spectMean.csv", delimiter=",")
    zeroTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/zero.csv", delimiter=",")
    energyTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/energy.csv", delimiter=",")
    #X_train = np.concatenate((mfccTrainData, spectTrainData, zeroTrainData, energyTrainData), axis=1)
    X_train = np.concatenate((mfccTrainData, energyTrainData), axis=1)
    print(X_train.shape)
    
    print("Loading test data...")
    mfccTestData = np.loadtxt("feature/acoustic/"+validateFolder+"/mfccMean.csv", delimiter=",")
    spectTestData = np.loadtxt("feature/acoustic/"+validateFolder+"/spectMean.csv", delimiter=",")
    zeroTestData = np.loadtxt("feature/acoustic/"+validateFolder+"/zero.csv", delimiter=",")
    energyTestData = np.loadtxt("feature/acoustic/"+validateFolder+"/energy.csv", delimiter=",")
    #X_test = np.concatenate((mfccTestData, spectTestData, zeroTestData, energyTestData), axis=1)
    X_test = np.concatenate((mfccTestData, energyTestData), axis=1)
    print(X_test.shape)

    numFiles = X_test.shape[0]
    numClasses = 30
    classScores = [[0]*numClasses for i in xrange(numFiles)]
    
    for classNum in range(1, numClasses+1):
        print("Loading training classfication for class {}...".format(classNum))
        numFiles = X_train.shape[0]
        fileIn = open("vine-venue-training-orderByName.txt", "r")
        venues = []
        i = 0
        for line in fileIn:
            if i == numFiles: break
            lineSplit = line.split("\t")
            v = int(lineSplit[1].replace("\n", ""))
            if v != classNum:
                v = 0
            venues.append(v)
            i += 1
        fileIn.close()
        Y_train = np.transpose(np.asarray(venues))

        numFiles = X_test.shape[0]
        fileIn = open("vine-venue-validation-orderByName.txt", "r")
        audioFiles = []
        venues = []
        i = 0
        for line in fileIn:
            if i == numFiles: break
            lineSplit = line.split("\t")
            v = int(lineSplit[1].replace("\n", ""))
            if v != classNum:
                v = 0
            audioFiles.append(int(lineSplit[0]))
            #venues.append(int(lineSplit[1].replace("\n", "")))
            venues.append(v)
            i += 1
        fileIn.close()

        Y_files = np.transpose(np.asarray(audioFiles))
        Y_truth = np.transpose(np.asarray(venues))

        print('Data Load Done.')

        print("Preparing output matrix...")
        Y_predicted = np.zeros([X_test.shape[0]])
        #print(Y_predicted.shape)

        print("Training classifier...")
        #model = svm.SVR(kernel='rbf', degree=3, gamma=0.1, shrinking=True, verbose=False, max_iter=-1)
        #model = svm.SVC(kernel='linear', C=1.0)
        model = svm.LinearSVC(C=1.0)
        model.fit(X_train, Y_train)

        print("Predicting classes for test data...")
        Y_predicted = np.asarray(model.predict(X_test))
        Y_scores = model.decision_function(X_test)
            
        print('SVM Train Done.')

        print("Prediction result:")
        print(classification_report(Y_truth, Y_predicted))

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

    numFiles = X_test.shape[0]
    fileIn = open("vine-venue-validation-orderByName.txt", "r")
    venues = []
    i = 0
    for line in fileIn:
        if i == numFiles: break
        lineSplit = line.split("\t")
        venues.append(int(lineSplit[1].replace("\n", "")))
        i += 1
    fileIn.close()
    Y_truth = np.transpose(np.asarray(venues))

    top1hit = 0
    top5hit = 0
    for i in range(numFiles):
        topClasses = []
        for j in range(5):
            topClasses.append(classScores[i].index(max(classScores[i])) + 1)
            classScores[i][classScores[i].index(max(classScores[i]))] = -999
        print i+1, ". Top 5:", topClasses, "Truth:", Y_truth[i]
        if topClasses[0] == int(Y_truth[i]):
            top1hit += 1
            top5hit += 1
        elif int(Y_truth[i]) in topClasses[1:]:
            top5hit += 1

    print "Final Result:"
    print "Top 1 hit:", top1hit, "/", numFiles
    print "Top 5 hit:", top5hit, "/", numFiles

if __name__ == '__main__':
    mySVM()
