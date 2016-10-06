import numpy as np
import cv2
from matplotlib import pyplot as plt

def chi2_distance(histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d

def getdes(img1):
    orb = cv2.ORB()
    img1 = cv2.imread(img1,0)
    img1 = img1.astype(np.uint8)
    kp1, des1 = orb.detectAndCompute(img1,None)
    return des1

def getorb():
    return cv2.ORB()

def myMatch(img1, img2):

    nam2 = img2.split('.')[0]
    img1 = cv2.imread(img1,0)          # queryImage
    img2 = cv2.imread(img2,0) # trainImage

    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    
    '''img1 = cv2.resize(img1, (299, 299)) 
    img2 = cv2.resize(img2, (299, 299)) '''
    
    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    

    # If no keypoints are found, return no matches
    if ((des1 is None) or (des2 is None)):
        return 0

    #kp2 = orb.detect(img2)

    '''op = open(nam2+".txt", 'w')
    print(nam2)
    for k in kp2:
        op.write("%d, %d, %d, %d, %d, %d, %d\n" % (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id))
    op.close()'''

    '''ip = open(nam2+".txt", 'r')
    kp2r = []
    for line in ip:
        x = line.split(', ')
        kp2r.append(cv2.KeyPoint(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), int(x[5]), int(x[6])))
    ip.close()

    kp2b, des2 = orb.compute(img2, kp2r)

    if ((des1 is None) or (des2 is None)):
        return 0'''

    #return chi2_distance(des1, des2)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    for i in range(len(matches)):
        if matches[i].distance > 70:
            return i
            #return ((len(matches) - i) / float(len(matches)) * 100)
    return len(matches)



def myMatch2(img1, img2):
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread(img1,0)          # queryImage
    img2 = cv2.imread(img2,0) # trainImage


    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    return len(good)

#myMatch('a.pgm', 'b.pgm')
