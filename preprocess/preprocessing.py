from lib import epl 
from lib import OSN
from lib import readout
from lib import plots
import numpy as np
import pickle 
import brian2 as b2
import brian2hears as b2h
import matplotlib.pyplot as plt

def perturb(trainingAudio, perRand = .1):
    '''perterb(trainingSim, perRand = .1): 
       trainingSim should be 1 by sensor number (100 or 72) 
       perRand is a percentage from 0 to 1
       returns perterbed trainingSim where a fraction of the elements are modified
    '''
    trainingSim = trainingAudio.copy()
    change_ndx = np.random.randint(0,trainingSim.shape[1],int(trainingSim.shape[1]*perRand))
    trainingSim[0,change_ndx] = np.random.randint(0,15, change_ndx.shape[0] )
    
    return trainingSim
    
def generatePerturbedTest(trainingSim, perRand = .1):
    '''
    generatePerturbed(trainingAudio)
    perRand (optional) default is .1 and must be a number between 0 and 1
    trainingAudio should be 1 by sensor number (100 or 72)
    returns perturbed testAudio which should be 10 by sensor number (100 or 72)
    '''
    testAudio = np.array([])
    x=10
    while x != 0:
        testAudio = np.concatenate(testAudio, perturb(trainingSim)) #changed from trainingAudio to trainingSim
        x-=1
    return testAudio
    
def chooseSound(trainingSpect, range_f, step_f, trainingIndex, scale):  #break into select sound and rescale input
    trainingAudio = trainingSpect[:,trainingIndex:trainingIndex+range_f:step_f].reshape((1,-1))
    #flatten list (np function) above to 1 row and 110 samples
    #trainingAudio = trainingAudio.flatten()
    #trainingAudio = trainingAudio.reshape((1,-1))
    #scale the training samples
    trainingAudio -= trainingAudio.min()
    trainingAudio *= (scale/trainingAudio.max())
    trainingAudio = trainingAudio.astype(np.int64)
    return trainingAudio

def selectSound(trainingSpect, range_f, step_f, trainingIndex):  # scale is not needed so I can delete it in the code and here
    trainingAudio = trainingSpect[:,trainingIndex:trainingIndex+range_f:step_f].reshape((1,-1))
    return trainingAudio
    
def rescaleInput(trainingAudio, scale, mini, rangei):  
    
    trainingAudio -= mini
    trainingAudio *= (scale/rangei)
    trainingAudio = trainingAudio.astype(np.int64)
    return trainingAudio
    
def NEWrescaleInput(trainingAudio, testAudio, scale): 
    mins = [trainingAudio.min(), testAudio.min()]
    maxs = [trainingAudio.max(), testAudio.max()]
    mini = min(mins)
    maxi = max(maxs)
    #rescaleInput(trainingAudio,scale)
    #rescaleInput(testAudio, scale)
    """
    trainingAudio -= mini
    trainingAudio *= (scale/maxi)
    trainingAudio = trainingAudio.astype(np.int64)
    testAudio -= mini
    testAudio *= (scale/maxi)
    testAudio = trainingAudio.astype(np.int64)
    """
    return mini# and testAudio


def spectArray(wav_sound, cfN=100, audio_type = ".wav", file_path = "C:/Users/micha/CODE/yes_sounds/" ):
    cfmin, cfmax = 50*b2.Hz, 3*b2.kHz
    # 72 chemosensors, but changing to 100 leads to finer frequencies and the network still works
    # 50 to 3000 Hz because normal human speech is within these frequencies
    cf = b2h.erbspace(cfmin, cfmax, cfN)
    sound = b2h.loadsound(file_path + wav_sound+audio_type)

    gfb = b2h.Gammatone(sound, cf)
    X = np.abs(gfb.process().T)
    X = np.log(X+.01)
    X += X.min()
   
    return X

# change name from spect to spectShow
def spectShow(X, zoom_x = None, zoom_y = None):  #parameters, frequency range, 
    X = X[:,zoom_x:zoom_y]
    plt.imshow(X,aspect='auto', origin='lower left')
    plt.set_cmap('gray_r')
    plt.ylabel('Frequency (channel number)')
    plt.xlabel('Time (index)')
    

def X_index(X, xcoords, xcoords2, scale = 10):
    trainingAudio = X[:,xcoords[0]].T/X.max()*scale #indexing needs revision
    trainingAudio = trainingAudio.astype(np.int64)

    testAudio = X[:,xcoords2[:]].T/X.max()*scale
    testAudio = testAudio.astype(np.int64)

#Sniff (hasn't been implemented in code yet)

#REMEMBER: add sniff to main code
def sniff(sound, learn_flag=0, nGammaPerSound=5, gPeriod=40):
    from lib import epl
    sensorInput = OSN.OSN_encoding(sound); 
    for j in range(0, nGammaPerSound):
        for k in range(0, gPeriod): 
            epl.update(sensorInput, learn_flag=learn_flag);
            pass; 
    epl.reset(); 
    

#add function to slice X array with parameters number of indices, 
#collect samples (training Sounds and test)
#normalize samples

def plotTrainedtoTarget(sMatrix): 
    bar1_x, bar2_x = [], [];      
    bar1_y, bar2_y = [], [];      
    nGamma = 5; 
    testSoundID = 0;              #sniff ID of test sound 
    for i in range(0, nGamma):
        bar2_y.append(sMatrix[nGamma*testSoundID+i][0]);
        bar1_y.append(sMatrix[nGamma*(testSoundID+1)+i][0]);
    w = 0.25        #width of bars
    fsize = 14;
    xticks = []
    xtick_labels = []
    for i in range(0, 5):    
        bar1_x.append(i-w/2);
        bar2_x.append(i+w/2);
        xticks.append(i)
        xtick_labels.append(str(i+1))
    plt.figure(figsize=(10, 6))
    fig = plt.subplot(111)
    bar1 = plt.bar(bar1_x, bar1_y, width = w, color = 'tab:orange', align='center', hatch="|")
    bar2 = plt.bar(bar2_x, bar2_y, width = w, color = 'tab:purple', align='center', hatch = "o")
    plt.xlim(-1, len(bar1_x)+0.2)
    plt.ylim(0.2, 1.02)
    plt.xticks(xticks, xtick_labels, fontsize=fsize)
    plt.yticks(fontsize=fsize); 
    fig.legend( (bar1, bar2), ('Trained', 'Target'), loc = 'upper left', fontsize=fsize)
    plt.ylabel('Similarity', fontsize=fsize); 
    plt.xlabel('Gamma Cycle', fontsize=fsize);
    plt.title("Similarity of trained to learned audio samples", fontsize=fsize); 
    plt.show()
    
def plotFigure3b(gammaCode, nMCs, nGamma = 5, alignment = '51', figsize = (6,20)): 
    plt.figure(1, figsize)
    #nGamma = 5; 
    #nMCs = len(trainingAudio[0]); 
    for i in range(0, nGamma):
        s = int(alignment+str(i+1))
        plt.subplot(s);
        if(i==0): 
            plt.title("Five gamma cycles of occluded toluene"); 
        for j in range(0, nMCs):
            #Learned Toluene
            gammaID = 1*nGamma + 9; 
            if gammaCode[gammaID][j] != 0:
                spikeTime = 20 - gammaCode[gammaID][j]; 
                plt.scatter(spikeTime, j, s= 8, marker = 'o', color = 'w', edgecolor='k')
            #Occluded Toluene
            testSample = 0; 
            gammaID = 10 + nGamma*testSample + i; 
            if gammaCode[gammaID][j] != 0:
                spikeTime = 20 - gammaCode[gammaID][j]; 
                plt.scatter(spikeTime, j, s= 2, marker = 'o', color = (0, 0, 1, 0.5)); 
            plt.ylim([-2,71])
            offset = 40*i; 
            plt.xticks([0, 4, 8, 12, 16, 20], [offset+0, offset+4, offset+8, offset+12, offset+16, offset+20]); 
            plt.yticks([15, 30, 45, 60]); 
    plt.xlabel('Timesteps');
    plt.ylabel('MC Index');
    plt.tight_layout()
    plt.show(); 

def my_plotFigure3b(gammaCode, epoch, nMCs, nGamma = 5, figsize = (6,20)): 
    plt.figure(1, figsize)
    alignment = '51'
    for i in range(0, nGamma):
        s = int(alignment+str(i+1))
        plt.subplot(s);
        for j in range(0, nMCs):
            #Learned Toluene
            gammaID = epoch*nGamma + i; 
            if gammaCode[gammaID][j] != 0:
                spikeTime = 20 - gammaCode[gammaID][j]; 
                plt.scatter(spikeTime, j, s= 8, marker = 'o', color = 'w', edgecolor='k')
            

def plot_images(digits, data, title, nrows=2, ncols=3):
    # display the images of the patterns (digits) to be learned

    fig, axs = plt.subplots(nrows, ncols, figsize=(8, nrows*4),
                        subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axs.flat):
        img = data[i]
        img = np.reshape(img, (8,8))
        ax.imshow(img, interpolation=None, cmap='viridis')
        ax.set_title(str(digits[i]))
    plt.tight_layout()
    fig.suptitle(title, fontsize=16)
    plt.show()
    
# jaccardSimilarity(trained[-1,i], tested[-1,j])
# i is the consecutive original digit. j is the consecutive test digit
# make an array from nested loop
def jaccardSimilarity(l1=[], l2=[]):
    """ computes Jaccard similarity"""
    list1 = []
    list2 = []
    for i in range(0, len(l1)):
        list1.append((i, l1[i]))
        list2.append((i, l2[i]))
    set1 = set(list1)
    set2 = set(list2)
    intersectionSize = len(set.intersection(set1, set2))
    unionSize = len(set.union(set1, set2))
    # print intersectionSize, unionSize;
    return round(intersectionSize/float(unionSize), 4)
def computeSimilarity(l1, l2):
    """ computes similarity index """
    return jaccardSimilarity(l1, l2)
def results(decisionMatrix, truthTable):
    true_negative = 0
    false_positive = 0
    comp = decisionMatrix == truthTable
    for i in comp:
        for j in i:
            if j == False:
                true_negative+=1
            else:
                false_positive+=1
    return (true_negative, false_positive)

def createTraining(trainingSounds, trainingIndex, cfN, range_f, step_f, scale):
    for sound in trainingSounds: 
        for i in trainingIndex:
            if i == trainingIndex[0]:
                trainingAudio = chooseSound(spectArray(sound, cfN=cfN), range_f, step_f, i, scale);
            else:
                trainingAudio = np.concatenate((trainingAudio, chooseSound(spectArray(sound,cfN=cfN), range_f, step_f, i, scale)))
    return trainingAudio

def createTesting(testSounds, cfN, testingIndex, colors, range_f, step_f, scale):
    testAudio = np.array([])
    for sound in testSounds:
        testingSpect = spectArray(sound, cfN=cfN)
        spectShow(testingSpect)

        plt.axvline(x=testingIndex[testSounds.index(sound)], c=colors[testSounds.index(sound)])
        plt.show()
        
        if testAudio.shape[0] == 0:
            testAudio = chooseSound(testingSpect, range_f, step_f, testingIndex[0],scale)
        else:
            testAudio = np.concatenate((testAudio,chooseSound(testingSpect, range_f, step_f, testingIndex[0],scale)))
    return testAudio
