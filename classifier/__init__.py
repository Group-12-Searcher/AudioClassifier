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
    mfccTrainData = np.loadtxt("feature/acoustic/train/mfccMean.csv", delimiter=",")
    spectTrainData = np.loadtxt("feature/acoustic/train/spectMean.csv", delimiter=",")
    zeroTrainData = np.loadtxt("feature/acoustic/train/zero.csv", delimiter=",")
    energyTrainData = np.loadtxt("feature/acoustic/train/energy.csv", delimiter=",")
    X_train = np.concatenate((mfccTrainData, spectTrainData, zeroTrainData, energyTrainData), axis=1)
    print(X_train.shape)

    print("Loading training classfication...")
    numFiles = X_train.shape[0]
    fileIn = open("vine-venue-training-orderByName.txt", "r")
    venues = []
    i = 0
    for line in fileIn:
        if i == numFiles: break
        venues.append(int(line.split("\t")[1].replace("\n", "")))
        i += 1
    fileIn.close()
    Y_train = np.transpose(np.asarray(venues))
    #print(Y_train)
    print(Y_train.shape)

    print("Loading test data...")
    mfccTestData = np.loadtxt("feature/acoustic/validate/mfccMean.csv", delimiter=",")
    spectTestData = np.loadtxt("feature/acoustic/validate/spectMean.csv", delimiter=",")
    zeroTestData = np.loadtxt("feature/acoustic/validate/zero.csv", delimiter=",")
    energyTestData = np.loadtxt("feature/acoustic/validate/energy.csv", delimiter=",")
    X_test = np.concatenate((mfccTestData, spectTestData, zeroTestData, energyTestData), axis=1)
    print(X_test.shape)

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
    np.savetxt("prediction.csv", Y_predicted, delimiter=",")
    #np.savetxt("prediction_scores.csv", Y_scores, delimiter=",")
    print("Done!")


if __name__ == '__main__':
    mySVM()
