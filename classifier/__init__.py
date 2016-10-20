import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.metrics import classification_report
import os

# Trian your own classifier.
# Here is the simple implementation of SVM classification.
def mySVM():
    os.chdir("../")

    print("Loading training data...")
    mfccTrainData = np.loadtxt("feature/acoustic/train_converted/mfccMean.csv", delimiter=",")
    spectTrainData = np.loadtxt("feature/acoustic/train_converted/spectMean.csv", delimiter=",")
    zeroTrainData = np.loadtxt("feature/acoustic/train_converted/zero.csv", delimiter=",")
    energyTrainData = np.loadtxt("feature/acoustic/train_converted/energy.csv", delimiter=",")
    X_train = np.concatenate((mfccTrainData, spectTrainData, zeroTrainData, energyTrainData), axis=1)
    print(X_train.shape)

    print("Loading training classfication...")
    numFiles = X_train.shape[0]
    fileIn = open("vine-venue-training-orderByName.txt", "r")
    venues = []
    i = 0
    for line in fileIn:
        if i == numFiles: break
        lineSplit = line.split("\t")
        venues.append(int(lineSplit[1].replace("\n", "")))
        i += 1
    fileIn.close()
    Y_train = np.transpose(np.asarray(venues))
    
    print("Loading test data...")
    mfccTestData = np.loadtxt("feature/acoustic/validate_converted/mfccMean.csv", delimiter=",")
    spectTestData = np.loadtxt("feature/acoustic/validate_converted/spectMean.csv", delimiter=",")
    zeroTestData = np.loadtxt("feature/acoustic/validate_converted/zero.csv", delimiter=",")
    energyTestData = np.loadtxt("feature/acoustic/validate_converted/energy.csv", delimiter=",")
    X_test = np.concatenate((mfccTestData, spectTestData, zeroTestData, energyTestData), axis=1)
    print(X_test.shape)

    numFiles = X_test.shape[0]
    fileIn = open("vine-venue-validation-orderByName.txt", "r")
    audioFiles = []
    venues = []
    i = 0
    for line in fileIn:
        if i == numFiles: break
        lineSplit = line.split("\t")
        audioFiles.append(int(lineSplit[0]))
        venues.append(int(lineSplit[1].replace("\n", "")))
        i += 1
    fileIn.close()

    Y_files = np.transpose(np.asarray(audioFiles))
    Y_truth = np.transpose(np.asarray(venues))

    print('Data Load Done.')

    print("Preparing output matrix...")
    Y_predicted = np.zeros([X_test.shape[0]])
    print(Y_predicted.shape)

    print("Training classifier...")
    #model = svm.SVR(kernel='rbf', degree=3, gamma=0.1, shrinking=True, verbose=False, max_iter=-1)
    model = svm.SVC(kernel='linear', C=1.0, degree=3, shrinking=True, verbose=False)
    model.fit(X_train, Y_train)

    print("Predicting classes for test data...")
    Y_predicted = np.asarray(model.predict(X_test))
    #Y_scores = model.decision_function(X_test)
        
    print('SVM Train Done.')

    print("Prediction result:")
    print(Y_predicted)
    
    print("Saving result...")
    print Y_files.shape, Y_predicted.shape, Y_truth.shape
    result = np.transpose(np.vstack((Y_files, Y_predicted, Y_truth)))
    np.savetxt("prediction.csv", result, delimiter=",")
    #np.savetxt("prediction_scores.csv", Y_scores, delimiter=",")
    print("Done!")


if __name__ == '__main__':
    mySVM()
