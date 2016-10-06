# import the necessary packages
import numpy as np
import csv
import cv2
import glob
import subprocess
import os
import threading
from mysift import myMatch

from sift.vwhist import vwHist

class Searcher:        
	def __init__(self, indexPath, semanticsPath, textPath, deeplearningPath, flags, searchTerm):
		# store our index path
		self.indexPath = indexPath
		self.semanticsPath = semanticsPath
		self.textPath = textPath
		self.deeplearningPath = deeplearningPath
		self.WEIGHT_CH = flags["ch"]
                self.WEIGHT_SEM = flags["vc"]
                self.WEIGHT_SIFT = flags["vk"]
                self.WEIGHT_TEXT = flags["tf"]
                self.searchTerm = searchTerm

                self.results = {}

        class myThread(threading.Thread):
            def __init__(self, searcher, dbImage):
                threading.Thread.__init__(self)
                self.searcher = searcher
                self.dbImage = dbImage
                
            def run(self):
                exe = 'match'
                im1 = "temp/query.pgm"
                k1 = "temp/query.key"
                im2 = self.dbImage+".pgm"
                k2 = self.dbImage+".key" 
                args = [exe, '-im1', im1, '-k1', k1, '-im2', im2, '-k2', k2]
                FNULL = open(os.devnull, 'w')

                p = subprocess.Popen(args, stdout=FNULL, stderr=subprocess.PIPE)
                line = p.stderr.readline()
                #print(line[0:-2]+" with "+self.dbImage)

                numMatches = line.split(' ')[1]
                d = float(numMatches)
                if self.searcher.results.get("dataset\\dataset\\"+self.dbImage+".jpg") is None:
                        self.searcher.results["dataset\\dataset\\"+self.dbImage+".jpg"] = d * self.searcher.WEIGHT_SIFT
                else:
                        self.searcher.results["dataset\\dataset\\"+self.dbImage+".jpg"] -= d * self.searcher.WEIGHT_SIFT                

	def search(self, queryFeatures, querySemantics, queryText, queryCategory, limit = 16):
                if self.WEIGHT_CH != 0:
                        # open the color histogram index file for reading
                        with open(self.indexPath) as f:
                                #print("Doing color histogram...")
                                # initialize the CSV reader
                                reader = csv.reader(f)

                                # loop over the rows in the index
                                for row in reader:
                                        # parse out the image ID and features, then compute the
                                        # chi-squared distance between the features in our index
                                        # and our query features
                                        features = [float(x) for x in row[1:]]
                                        d = self.chi2_distance(features, queryFeatures)
                                        
                                        # now that we have the distance between the two feature
                                        # vectors, we can udpate the self.results dictionary -- the
                                        # key is the current image ID in the index and the
                                        # value is the distance we just computed, representing
                                        # how 'similar' the image in the index is to our query
                                        self.results[row[0]] = d*10 * self.WEIGHT_CH
				
			# close the reader
			f.close()


                if self.WEIGHT_SEM != 0:
                        # open the semantics index file for reading
                        with open(self.semanticsPath) as f:
                                #print("Doing semantics...")
                                # initialize the CSV reader
                                reader = csv.reader(f)

                                # loop over the rows in the index
                                for row in reader:
                                        # parse out the image ID and features, then compute the
                                        # chi-squared distance between the features in our index
                                        # and our query features
                                        features = [float(x) for x in row[1:]]
                                        '''
                                        features = []
                                        for x in row[1:]:
                                                print(x)
                                                print("---")
                                                features.append(float(x))
                                        '''

                                                                
                                        d = self.chi2_distance(features, querySemantics)
                                        
                                        # now that we have the distance between the two feature
                                        # vectors, we can udpate the results dictionary -- the
                                        # key is the current image ID in the index and the
                                        # value is the distance we just computed, representing
                                        # how 'similar' the image in the index is to our query
                                        if self.results.get(row[0]) is None:
                                                self.results[row[0]] = d * self.WEIGHT_SEM
                                        else:
                                                self.results[row[0]] += d * self.WEIGHT_SEM
                                        
                                # close the reader
                                f.close()

                if self.WEIGHT_SIFT != 0:
                        # Do sift matching
                        '''
                        os.chdir("sift")
                        threads = []
                        for f in glob.glob("*.key"):
                                dbImage = f.split('/')[-1].split('.')[0]
                                t = self.myThread(self, dbImage)
                                t.start()
                                threads.append(t)
                        for t in threads:
                                t.join()'''
                                
                        ### My own experimental sift ###
                        #print("Doing sift matching...")
                        os.chdir("sift")
                        im1 = "temp/query.pgm"
                        hist1 = vwHist(im1, False)
                        for f in glob.glob("*.pgm"):
                                dbImage = f.split('/')[-1].split('.')[0]
                                im1 = "temp/query.pgm"
                                im2 = dbImage+".pgm"
                                hist2 = []
                                histFile = open(dbImage+'.txt', 'r')
                                for x in histFile:
                                        hist2.append(float(x))
                                d = self.chi2_distance(hist1, hist2)
                                #d = myMatch(im1, im2)
                                req_img_name = "dataset\\dataset\\"+dbImage+".jpg"
                                if self.results.get(req_img_name) is None:
                                        self.results[req_img_name] = d * self.WEIGHT_SIFT
                                else:
                                        self.results[req_img_name] += d * self.WEIGHT_SIFT
                        os.chdir("..")

                if self.WEIGHT_TEXT != 0 or self.searchTerm != "":
                        # open the text index file for reading
                        with open(self.textPath) as f:
                                #print("Doing text search...")
                                # initialize the CSV reader
                                reader = csv.reader(f)
                                # Loop over all the tagged words for query image
                                # print(queryText)
                                if (self.searchTerm != ""):
                                        # print("Search term is supplied: %s" % self.searchTerm)
                                        # loop over the rows in the index
                                        for row in reader:
                                                # print(row[0])
                                                if (row[0] == self.searchTerm):
                                                        # print("Match detected for word: %s" % word)
                                                        # loop over all matching images
                                                        for match_img in row[1:]:
                                                                req_name = "dataset\\dataset\\" + match_img
                                                                if (os.path.exists(req_name)):
                                                                        if (self.results.get(req_name) is not None):
                                                                                # print("Modifying value in existing result set for %s" % req_name)
                                                                                self.results[req_name] -= self.WEIGHT_TEXT*100
                                                                                # print(self.results[req_name])
                                                                        else:
                                                                                # print("Adding this file: %s" % req_name)
                                                                                self.results[req_name] = self.WEIGHT_TEXT*100
                                                                                # print(self.results[req_name])
                                                        break # go to next word
                                        
                                if (queryText != None):
                                        # print("Query image has text tags.")
                                        #print(queryText)
                                        for word in queryText:
                                                # loop over the rows in the index
                                                for row in reader:
                                                        if (row[0] == word):
                                                                # print("Match detected for word: %s" % word)
                                                                # loop over all matching images
                                                                for match_img in row[1:]:
                                                                        req_name = "dataset\\dataset\\" + match_img
                                                                        if (os.path.exists(req_name)):
                                                                                if (self.results.get(req_name) is not None):
                                                                                        # print("Modifying value in existing result set for %s" % req_name)
                                                                                        self.results[req_name] -= self.WEIGHT_TEXT*100
                                                                                        # print(self.results[req_name])
                                                                                else:
                                                                                        # print("Adding this file: %s" % req_name)
                                                                                        self.results[req_name] = self.WEIGHT_TEXT*100
                                                                                        # print(self.results[req_name])
                                                                break # go to next word
                                                
                                # close the reader
                                f.close()

                '''if self.WEIGHT_DL != 0:
                        # open the deep-learning index file for reading
                        with open(self.deeplearningPath) as f:
                                print("Doing deep-learning matching...")
                                # initialize the CSV reader
                                reader = csv.reader(f)

                                for row in reader:
                                        if row[0] == queryCategory:
                                                scores = row[1:]
                                                break

                                categs = open("category_names.txt", 'r')
                                catScores = {}
                                i = 0
                                for cat in categs:
                                        catScores[cat.split('\n')[0]] = float(scores[i])
                                        i += 1
                                categs.close()
                                        
                                index_categories = open("index_categories.csv", 'r')
                                catReader = csv.reader(index_categories)
                                imageCat = {}
                                for row in catReader:
                                        imageCat[row[0]] = row[1:]
                                index_categories.close()
                                
                                for imgPath in glob.glob("dataset\\dataset\\*.jpg"):
                                        img = imgPath.split('\\')[-1]
                                        cats = imageCat[img]
                                        if len(cats) == 1:
                                                score = catScores[cats[0]]*100 * self.WEIGHT_DL
                                        else:
                                                score = 0
                                                for c in cats:
                                                        if catScores[c] > score:
                                                                score = catScores[c]
                                                score = score*100 * self.WEIGHT_DL
                                        self.results[imgPath] -= score
                                        #print(img, score)
                        f.close()'''
                
                # print(len(self.results))

		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		self.results = sorted([(v, k) for (k, v) in self.results.items()])
		
		# return our (limited) self.results
		topImages = []
		for r in self.results[:limit]:
                        topImages.append(r[1].split('\\')[-1])
                        
                index_categories = open("index_categories.csv", 'r')
                catReader = csv.reader(index_categories)
                topImagesCats = [0]*16
                for row in catReader:
                        if row[0] in topImages:
                                topImagesCats[topImages.index(row[0])] = row[1:]
                index_categories.close()
                
                i = 1
                for cats in topImagesCats:
                        #print("{}. {}: {}".format(i, topImages[i-1], cats))
                        i += 1
                #print("")
		return self.results[:limit]

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d
