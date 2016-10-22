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
    X_train = np.concatenate((mfccTrainData, spectTrainData, zeroTrainData, energyTrainData), axis=1)
    #X_train = np.concatenate((mfccTrainData, spectTrainData), axis=1)
    #X_train = spectTrainData
    print(X_train.shape)
    
    print("Loading test data...")
    mfccTestData = np.loadtxt("feature/acoustic/"+validateFolder+"/mfccMean.csv", delimiter=",")
    spectTestData = np.loadtxt("feature/acoustic/"+validateFolder+"/spectMean.csv", delimiter=",")
    zeroTestData = np.loadtxt("feature/acoustic/"+validateFolder+"/zero.csv", delimiter=",")
    energyTestData = np.loadtxt("feature/acoustic/"+validateFolder+"/energy.csv", delimiter=",")
    X_test = np.concatenate((mfccTestData, spectTestData, zeroTestData, energyTestData), axis=1)
    #X_test = np.concatenate((mfccTestData, spectTestData), axis=1)
    #X_test = spectTestData
    print(X_test.shape)

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

    numFiles = X_test.shape[0]
    fileIn = open("vine-venue-validation-orderByName.txt", "r")
    venues = []
    i = 0
    for line in fileIn:
        if i == numFiles: break
        lineSplit = line.split("\t")
        f = int(lineSplit[0])
        v = int(lineSplit[1].replace("\n", ""))
        for j in range(numGroups):
            if v in categories[j]:
                venues.append(j+1)
                break
        i += 1
    fileIn.close()
    Y_truth = np.transpose(np.asarray(venues))

    print('Data Load Done.')

    print("Preparing output matrix...")
    Y_predicted = np.zeros([X_test.shape[0]])

    print("Training the 6-groups classifier...")
    model = svm.SVC(kernel='linear', C=0.00001, degree=3, shrinking=True, verbose=False)
    #model = svm.LinearSVC(C=0.00001)
    model.fit(X_train, Y_train)

    print("Predicting group for test data...")
    Y_predicted = np.asarray(model.predict(X_test))
    grouphits = 0
    for i in range(numFiles):
        if Y_predicted[i] == Y_truth[i]:
            grouphits += 1
    
    print('SVM Train Done.')
    print("Classification report:")
    print(classification_report(Y_truth, Y_predicted))

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

    totalhits = 0
    topRankings = []
    for i in range(numGroups):
        print "Training classifier for group", i+1
        if len(X_test_list[i]) == 0:
            print "Group is empty!"
            continue
        
        model = svm.SVC(kernel='linear', C=0.0001, degree=3, shrinking=True, verbose=False)
        model.fit(X_train_list[i], Y_train_list[i])
        model_linear = svm.LinearSVC(C=0.00001)
        model_linear.fit(X_train_list[i], Y_train_list[i])
        
        Y_predicted = np.asarray(model.predict(X_test_list[i]))
        Y_scores = np.asarray(model_linear.decision_function(X_test_list[i]))
        print('SVM Train Done.')
        print("Classification report:")
        print(classification_report(Y_truth_list[i], Y_predicted))         

        k = Y_predicted.shape[0]
        hit = 0
        for j in range(k):
            if Y_truth_list[i][j] == Y_predicted[j]:
                hit += 1
            ranking = []
            ranking.append(int(Y_predicted[j]))

            scores = Y_scores[j].tolist()
            scores[categories[i].index(Y_predicted[j])] = -999
            for m in range(len(categories[i])-1):
                v = categories[i][scores.index(max(scores))]
                ranking.append(v)
                scores[categories[i].index(v)] = -999
            print j+1, "- Top venues are:", ranking

            while len(ranking) < numGroups:
                ranking.append(0)
            topRankings.append(ranking)
            
        print "Number of hits:", hit, "/", k
        totalhits += hit
        
        print("Saving result...")
        prediction = np.transpose(np.vstack((np.asmatrix(Y_files_list[i]), Y_predicted, np.asmatrix(Y_truth_list[i]))))
        np.savetxt("prediction_group"+str(i+1)+".csv", prediction, delimiter=",")
        np.savetxt("prediction_group"+str(i+1)+"_scores.csv", Y_scores, delimiter=",")

    np.savetxt("toprankings.csv", np.array(topRankings), delimiter=",")
        
    print "\nGrand total number of hits:", totalhits, "/ 900"
    print "Number of group hits:", grouphits, "/ 900"
    
    '''
    print("Saving result...")
    print Y_files.shape, Y_predicted.shape, Y_truth.shape
    result = np.transpose(np.vstack((Y_files, Y_predicted, Y_truth)))
    np.savetxt("prediction.csv", result, delimiter=",")
    np.savetxt("prediction_scores.csv", Y_scores, delimiter=",")   
    '''
    
    print("Done!")

if __name__ == '__main__':
    mySVM()
