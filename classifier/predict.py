import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.metrics import classification_report
import os
from sklearn.externals import joblib
import time

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
    X_test = np.concatenate((mfccTestData, spectTestData, zeroTestData, energyTestData), axis=1)
    #X_test = np.concatenate((spectTestData, mfccTestData), axis=1)

    veryNoisy = [1,21,22,26]
    sometimesNoisy = [3,4,9,14,20]
    notVeryNoisy = [5,6,8,19,28]
    mostlyQuiet = [11,16,24,29,30]
    sports = [10,12,13,15,23,27]
    music = [2,7,17,18,25]
    categories = [veryNoisy, sometimesNoisy, notVeryNoisy, mostlyQuiet, sports, music]
    
    numGroups = len(categories)
    X_test_list = [[],[],[],[],[],[]]
    Y_truth_list = [[],[],[],[],[],[]]
    Y_files_list = [[],[],[],[],[],[]]
    
    numFiles = X_test.shape[0]
    fileIn = open("vine-venue-validation-orderByName.txt", "r")
    groups = []
    venues = []
    i = 0
    for line in fileIn:
        if i == numFiles: break
        lineSplit = line.split("\t")
        f = int(lineSplit[0])
        v = int(lineSplit[1].replace("\n", ""))
        for j in range(numGroups):
            if v in categories[j]:
                groups.append(j+1)
                venues.append(v)
                break
        i += 1
    fileIn.close()
    Y_groupTruth = np.transpose(np.asarray(groups))
    Y_truth = np.transpose(np.asarray(venues))
    print('Data Load Done.')

    Y_predicted = np.zeros([X_test.shape[0]])
    model = joblib.load('classifier/savedmodel.pkl')

    print("Predicting group for test data...")
    Y_predicted = np.asarray(model.predict(X_test))
    grouphits = 0
    for i in range(numFiles):
        if Y_predicted[i] == Y_groupTruth[i]:
            grouphits += 1

    groupPrediction = []
    for i in range(Y_predicted.shape[0]):
        groupPrediction.append(Y_predicted[i])
    #print groupPrediction
    
    print('SVM Train Done.')

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
    topRankings = [[],[],[],[],[],[]]
    for i in range(numGroups):
        #print "Retrieving classifier for group", i+1
        if len(X_test_list[i]) == 0:
            #print "Group is empty!"
            continue
        
        model = joblib.load('classifier/savedmodel' + str(i) + '.pkl')
        model_linear = joblib.load('classifier/savedmodellinear' + str(i) + '.pkl')
        
        Y_predicted = np.asarray(model.predict(X_test_list[i]))
        Y_scores = np.asarray(model_linear.decision_function(X_test_list[i]))       

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
            #print j+1, "- Top venues are:", ranking

            while len(ranking) < numGroups:
                ranking.append(0)
            topRankings[i].append(ranking)
        #print "Number of hits:", hit, "/", k
        totalhits += hit

    finalRanking = []
    k = len(groupPrediction)
    for i in range(k):
        #print i+1, groupPrediction[i]-1
        #print topRankings[Y_predicted[i]-1]
        finalRanking.append(topRankings[groupPrediction[i]-1][0])
        del topRankings[groupPrediction[i]-1][0]

    #print "\nGrand total number of hits:", totalhits, "/ ", X_test.shape[0]
    #print "Number of group hits:", grouphits, "/ ", X_test.shape[0]

    hitReport = [[0,0] for i in range(30)]
    for i in range(X_test.shape[0]):
        prediction = finalRanking[i][0]
        truth = Y_truth[i]
        #print prediction, truth
        if prediction == truth:
            hitReport[truth-1][0] += 1
            hitReport[truth-1][1] += 1
        else:
            predictionGroup = 0
            truthGroup = 0
            for j in range(numGroups):
                if prediction in categories[j]:
                    predictionGroup = j+1
                    break
            for j in range(numGroups):
                if truth in categories[j]:
                    truthGroup = j+1
                    break
            if predictionGroup == truthGroup:
                hitReport[truth-1][1] += 1

    #print "Number of hits / groups hits per venue:"
    #for i in range(30):
    #    print i+1, "-", hitReport[i]
    
    #print("Done!")

    if (isMain):
        print finalRanking
    else:
        index = -1
        with open('presentation/nameToIndex.txt', 'r') as mapFile:
            for line in mapFile:
                parts = line.split(" ")
                extractedName = parts[0]
                extractedIndex = parts[1]
                if (extractedName == name):
                    index = int(extractedIndex)
                    return finalRanking[index]  

if __name__ == '__main__':
    start_time = time.time()
    predictSVM("", True)
    #print(predictSVM("1.wav"))
    print("Time taken : " + str(time.time() - start_time))
