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
    validateFolder = "presentation"

    print("Loading training data...")
    mfccTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/mfccMean.csv", delimiter=",")
    spectTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/spectMean.csv", delimiter=",")
    zeroTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/zero.csv", delimiter=",")
    energyTrainData = np.loadtxt("feature/acoustic/"+trainFolder+"/energy.csv", delimiter=",")
    X_train = np.concatenate((mfccTrainData, spectTrainData, zeroTrainData, energyTrainData), axis=1)
    #X_train = np.concatenate((spectTrainData, mfccTrainData), axis=1)

    print("Loading test data...")
    mfccTestData = np.loadtxt(validateFolder+"/mfccMean.csv", delimiter=",")
    spectTestData = np.loadtxt(validateFolder+"/spectMean.csv", delimiter=",")
    zeroTestData = np.loadtxt(validateFolder+"/zero.csv", delimiter=",")
    energyTestData = np.loadtxt(validateFolder+"/energy.csv", delimiter=",")
    X_test = np.concatenate((mfccTestData, spectTestData, zeroTestData, energyTestData), axis=1)
    #X_test = np.concatenate((mfccTestData, spectTestData), axis=1)
    #X_test = spectTestData

    veryNoisy = [1,21,22,26]
    sometimesNoisy = [3,4,9,14,20]
    notVeryNoisy = [5,6,8,19,28]
    mostlyQuiet = [11,16,24,29,30]
    sports = [10,12,13,15,23,27]
    music = [2,7,17,18,25]
    categories = [veryNoisy, sometimesNoisy, notVeryNoisy, mostlyQuiet, sports, music]
    
    numGroups = len(categories)
    X_train_list = [[],[],[],[],[],[]]
    Y_train_list = [[],[],[],[],[],[]]
    X_test_list = [[],[],[],[],[],[]]
    Y_truth_list = [[],[],[],[],[],[]]
    Y_files_list = [[],[],[],[],[],[]]

    numFiles = X_train.shape[0]
    fileIn = open("vine-venue-training-orderByName.txt", "r")
    venues = []
    i = 0
    for line in fileIn:
        if i == numFiles: break
        lineSplit = line.split("\t")
        v = int(lineSplit[1].replace("\n", ""))
        for j in range(numGroups):
            if v in categories[j]:
                venues.append(j+1)
                X_train_list[j].append(X_train[i])
                Y_train_list[j].append(v)
                break
        i += 1
    fileIn.close()
    Y_train = np.transpose(np.asarray(venues))
    print('Data Load Done.')

    numFiles = X_test.shape[0]
    

    print("Training the 6-groups classifier...")
    model = svm.SVC(kernel='linear', C=0.00001, degree=3, shrinking=True, verbose=False)
    #model = svm.LinearSVC(C=0.00001)
    modelToSave = model.fit(X_train, Y_train)
    print('SVM Train Done.')
    joblib.dump(modelToSave, 'classifier/savedmodel.pkl')
    print('SVM Model saved.')
    Y_predicted = np.zeros([X_test.shape[0]])
    Y_predicted = np.asarray(model.predict(X_test))

    fileIn = open("vine-venue-validation-orderByName.txt", "r")
    i = 0
    for line in fileIn:
        if i == numFiles: break
        lineSplit = line.split("\t")
        f = int(lineSplit[0])
        v = int(lineSplit[1].replace("\n", ""))
        
        j = Y_predicted[i] - 1
        X_test_list[j].append(X_test[i])
        Y_truth_list[j].append(v)
        Y_files_list[j].append(f)
        i += 1
    fileIn.close()
        
    for i in range(numGroups):
        print "Training classifier for group", i+1
        if len(X_test_list[i]) == 0:
            print "Group is empty!"
            continue
        
        model = svm.SVC(kernel='linear', C=0.0001, degree=3, shrinking=True, verbose=False)
        modelToSave = model.fit(X_train_list[i], Y_train_list[i])
        joblib.dump(modelToSave, 'classifier/savedmodel' + str(i) + '.pkl')
        model_linear = svm.LinearSVC(C=0.00001)
        linearModel = model_linear.fit(X_train_list[i], Y_train_list[i])
        joblib.dump(linearModel, 'classifier/savedmodellinear' + str(i) + '.pkl')

if __name__ == '__main__':
    trainMySVM()
