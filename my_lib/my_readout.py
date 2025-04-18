from my_lib import a_library
def jaccardSimilarity(l1 = [], l2 =[]):
    list1 = [];
    list2 = [];
    for i in range(0, len(l1)):
        list1.append((i, l1[i]));
        list2.append((i, l2[i]));
    set1 = set(list1);
    set2 = set(list2);
    intersectionSize = len(set.intersection(set1, set2));
    unionSize = len(set.union(set1, set2));
    return intersectionSize/float(unionSize);

def computeSimilarity(l1, l2):
    return jaccardSimilarity(l1, l2);

def findLearnedGammaCode(gammaCode, nOdors, nGammaPerLearning):
    """
    returns gammaCode at index 9, 19, 29, 39 ... , 99
    what I want: 54, 59, 64, 69, 74, ..., 99
    """
    learnedGammaCode = [];  
    for i in range(1, nOdors+1):
        labelGammaID = nGammaPerLearning * nOdors + nGammaPerLearning*i - 1; 
        learnedGammaCode.append(gammaCode[labelGammaID]); 
    return learnedGammaCode; 

def findSImatrixGamma(gammaCode, learnedGammaCode, nOdors, testStartID):
    SImatrixGamma = []; 
    k = 0; 
    for i in range(testStartID, len(gammaCode)):
        SImatrixGamma.append([]); 
        for j in range(0, nOdors): 
            similarity = computeSimilarity(gammaCode[i], learnedGammaCode[j]);
            SImatrixGamma[k].append(round(similarity, 2));
        k+=1; 
    return SImatrixGamma; 

def readout(gammaCode, nOdors, nTestPerOdor, nGammaPerLearning = 5):
  
    learnedGammaCode = findLearnedGammaCode(gammaCode, nOdors, nGammaPerLearning);
    testStartID = nOdors*nGammaPerLearning + nOdors*nGammaPerLearning;     #learning + labeling 
    
    SImatrix = findSImatrixGamma(gammaCode, learnedGammaCode, nOdors, testStartID);
    
    return SImatrix