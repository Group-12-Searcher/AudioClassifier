import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.metrics import classification_report
import os
from sklearn.externals import joblib

# Train your own classifier.
# Here is the simple implementation of SVM classification.
def trainMySVM():
    os.chdir("../")
    trainFolder = "train_converted"

    print("Loading training data...")
    mfccTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/mfccMean.csv", delimiter=",")
    spectTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/spectMean.csv", delimiter=",")
    zeroTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/zero.csv", delimiter=",")
    energyTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/energy.csv", delimiter=",")
    #X_train = np.concatenate((mfccTrainData, spectTrainData, zeroTrainData, energyTrainData), axis=1)
    X_train = np.concatenate((spectTrainData, mfccTrainData), axis=1)

    numClasses = 30
    
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
        print('Data Load Done.')

        print("Training classifier...")
        #model = svm.SVR(kernel='rbf', degree=3, gamma=0.1, shrinking=True, verbose=False, max_iter=-1)
        #model = svm.SVC(kernel='linear', C=1.0)
        model = svm.LinearSVC(C=0.0000001)
        modelToSave = model.fit(X_train, Y_train)
        print('SVM Train Done.')
        joblib.dump(modelToSave, 'classifier/savedmodel' + str(classNum) + '.pkl')
        print('SVM Model saved.')
        
        '''print("Saving result...")
        print Y_files.shape, Y_predicted.shape, Y_truth.shape
        result = np.transpose(np.vstack((Y_files, Y_predicted, Y_truth)))
        np.savetxt("prediction.csv", result, delimiter=",")
        np.savetxt("prediction_scores.csv", Y_scores, delimiter=",")'''
        
    print("Done!")

if __name__ == '__main__':
    trainMySVM()
