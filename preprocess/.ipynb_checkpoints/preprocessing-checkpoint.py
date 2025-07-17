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

def selectSound(trainingSpect, range_f, step_f, trainingIndex):  # scale is not needed so I can delete it in the code and here
    trainingAudio = trainingSpect[:,trainingIndex:trainingIndex+range_f:step_f].reshape((1,-1))
    return trainingAudio
    
def rescaleInput(trainingAudio, scale):  
    mini = np.min(trainingAudio)
    maxi = np.max(trainingAudio)
    if maxi == 0:
        return trainingAudio
    trainingAudio -= mini
    trainingAudio = trainingAudio * (scale / (maxi-mini))
    trainingAudio = np.clip(np.round(trainingAudio), 0, scale).astype(np.int64)
    return trainingAudio


def rescaleInputBoth(trainingAudio, testAudio, scale): 
    # change name for all modalities, not just audio
    mins = [trainingAudio.min(), testAudio.min()]
    maxs = [trainingAudio.max(), testAudio.max()]
    mini = np.min(mins)
    maxi = np.max(maxs)
    
    trainingAudio -= mini
    trainingAudio = trainingAudio * (scale / (maxi-mini))
    trainingAudio = np.clip(np.round(trainingAudio), 0, scale).astype(np.int64)

    testAudio -= mini
    testAudio = testAudio * (scale / (maxi-mini))
    testAudio = np.clip(np.round(testAudio), 0, scale).astype(np.int64)
    
    return trainingAudio, testAudio


def spectArray(wav_sound, cfN=100, audio_type = ".wav", file_path = "C:/Users/micha/A-Neuromorphic-App-to-Keyword-Recog-main/A-Neuromorphic-App-to-Keyword-Recog-main/yes_sounds/" ):
    cfmin, cfmax = 50*b2.Hz, 3*b2.kHz
    # 72 chemosensors, but changing to 100 leads to finer frequencies and the network still works
    # 50 to 3000 Hz because normal human speech is within these frequencies
    cf = b2h.erbspace(cfmin, cfmax, cfN)
    sound = b2h.loadsound(file_path + wav_sound+audio_type)
    # print(sound.samplerate)
    
    #gfb = b2h.Gammatone(b2h.LowPass(sound,cfmax/2), cf)
    gfb = b2h.Gammatone(sound, cf)

    X = np.abs(gfb.process().T)
    X = np.log(X+.01)
    X += X.min()
   
    return X

# change name from spect to spectShow
def spectShow(X, zoom_y0 = None, zoom_y1 = None):  #parameters, frequency range, 
    X = X[:,zoom_y0:zoom_y1]
    plt.imshow(X,aspect='auto', origin='lower')
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

def chooseSound(trainingSpect, range_f, step_f, trainingIndex, scale):  #break into select sound and rescale input
    trainingAudio = trainingSpect[:,trainingIndex:trainingIndex+range_f:step_f].reshape((1,-1))
    trainingAudio = rescaleInput(trainingAudio, scale)
    return trainingAudio

def createTraining(trainingSounds, trainingIndex, cfN, range_f, step_f, scale):
    for sound in trainingSounds: 
        #visualization of indices and spectograms
        trainingSpect = spectArray(sound, cfN=cfN)
        [plt.axvline(x=ind) for ind in trainingIndex]
        spectShow(trainingSpect)  
        plt.show()
        for ind in trainingIndex:
            if ind == trainingIndex[0]:
                trainingAudio = chooseSound(spectArray(sound, cfN=cfN), range_f, step_f, ind, scale);
            else:
                trainingAudio = np.concatenate((trainingAudio, chooseSound(spectArray(sound,cfN=cfN), range_f, step_f, ind, scale)))
    return trainingAudio

def createTesting(testSounds, cfN, testingIndex, colors, range_f, step_f, scale):
    testAudio = np.array([])
    for sound in testSounds:
        #visualization of indices and spectograms
        testingSpect = spectArray(sound, cfN=cfN)
        plt.axvline(x=testingIndex[testSounds.index(sound)], c=colors[testSounds.index(sound)])
        spectShow(testingSpect)
        plt.show()
        
        for ind in testingIndex:
            if ind == testingIndex[0]:
                testAudio = chooseSound(testingSpect, range_f, step_f, ind,scale)
            else:
                testAudio = np.concatenate((testAudio,chooseSound(testingSpect, range_f, step_f, ind,scale)))
    
    return testAudio


def learningBarFigure(sMatrix, plotTitles, testOdorID = 0, noiseLevel = 0, nGamma = 5): 
    nOdors = len(sMatrix[0]) #for future so that number of bars displayed could be a parameter
    nTestPerOdor = len(sMatrix)/(nOdors*nGamma);       # = 120 / (6*5) = 4      

    bar1_x, bar2_x, bar3_x, bar4_x, bar5_x, bar6_x = [], [], [], [], [], [];      
    bar1_y, bar2_y, bar3_y, bar4_y, bar5_y, bar6_y = [], [], [], [], [], [];      
    
    odorID = (testOdorID*nTestPerOdor)*nGamma
    for i in range(0, nGamma):
        bar1_y.append(sMatrix[odorID+i][0]);
        bar2_y.append(sMatrix[odorID+i][1]);
        bar3_y.append(sMatrix[odorID+i][2]);
        bar4_y.append(sMatrix[odorID+i][3]);
        bar5_y.append(sMatrix[odorID+i][4]);
        bar6_y.append(sMatrix[odorID+i][5]);

    w = 0.15        #width of bars
    fsize = 14;
    xticks = []
    xtick_labels = []
    for i in range(0, nGamma):    
        bar1_x.append(i-2*w);
        bar2_x.append(i-w);
        bar3_x.append(i);
        bar4_x.append(i+w); 
        bar5_x.append(i+2*w);     
        bar6_x.append(i+3*w);     

        xticks.append(i)
        xtick_labels.append(str(i+1))
    plt.figure(2, figsize=(10, 5))
    fig = plt.subplot(111)
    opacity = 0.5; 
    bar1 = plt.bar(bar1_x, bar1_y, width = w, color = 'blue', alpha = opacity, align='center')
    bar2 = plt.bar(bar2_x, bar2_y, width = w, color = 'violet', alpha = opacity, align='center')
    bar3 = plt.bar(bar3_x, bar3_y, width = w, color = 'red', alpha = opacity, align='center')
    bar4 = plt.bar(bar4_x, bar4_y, width = w, color = 'orange', alpha=opacity, align='center')
    bar5 = plt.bar(bar5_x, bar5_y, width = w, color = 'green', alpha=opacity, align='center')
    bar6 = plt.bar(bar6_x, bar6_y, width = w, color = 'yellow', alpha=opacity, align='center')

    plt.xlim(-1, len(bar1_x)+0.2)
    plt.ylim(0, 1.02)
    plt.xticks(xticks, xtick_labels, fontsize=fsize)
    plt.yticks(fontsize=fsize); 
    fig.legend( (bar1, bar2, bar3, bar4, bar5, bar6), plotTitles, loc = 'upper left', fontsize=fsize)
    plt.ylabel('Similarity', fontsize=fsize); 
    plt.xlabel('Gamma Cycles', fontsize=fsize); 
    plt.title("learningBarFigure")
    plt.show()

def rasterSimilarityPlot(gammaCode, sMatrix, plotTitles, nMCs = 72, nOdors = 10, nGamma = 5): 
    """
    raster and similarity plot (my plotFigure4d)
    
    """
    #formatting
    fig, axs = plt.subplots(figsize=(18,9));
    fig.suptitle("Figure 4d", fontsize=16); 
    fsize=12;
    fsize2=16;  
    plt.subplots_adjust(hspace=0.45, wspace=0.25);

    #indexing for 
    sampleNumbers = [0, 0, 0, 0, 0, 9, 0, 0, 0, 1]; 
    #plotTitles = ['Acetaldehyde', 'Acetone', 'Ammonia', 'Benzene', 'Butanol', 'Carbon Monoxide', 'Ethylene', 'Methane', 'Methanol', 'Toluene'];
    #plotTitles = [0, 2, 4, 5, 7, 9]
    #Order of stim presentation in simulation: 
    #Toluene, Benzene, Methane, Carbon Monoxide, Ammonia, Acetone, Acetaldehyde, Methanol, Butanol, Ethylene
    #rasterOrderOdors = [6, 5, 4, 1, 8, 3, 9, 2, 7, 0];         #alphabetical ordering for plots
    rasterOrder = np.arange(nOdors)
    #plotTitles = [plotTitles[i] for i in rasterOrder ]
    nTestPerOdor = len(sMatrix)/(nOdors*nGamma);       # = 500 / (10*5) = 10      
    for i in range(0,nOdors):
        
        #Raster plot
        plt.subplot(5, 4, 2*i+1);
        odorID = i;  #rasterOrderOdors[i];  
        sniffID = nTestPerOdor*odorID;# +sampleNumbers[odorID];
        for j in range(0, nGamma): 
            gammaID = nGamma*nOdors*2 + nGamma*sniffID + j; 
            for k in range(0, nMCs):
                #print([gammaID, k])
                if(gammaCode[gammaID][k] != 0): 
                    spikeTime = 20 - gammaCode[gammaID][k] + 40*j;
                    plt.scatter(spikeTime, k, color='k', s=2);
                    #gammaID: 400, 401, 402, 403, 404,  355, 356, 357, 358, 359
                    
                    
        #formatting
        plt.title(plotTitles[i], fontsize=fsize2); 
        plt.yticks(range(nMCs/4, nMCs+1, nMCs/4), fontsize=fsize);
        if(i>=8): 
            plt.xticks([0, 40, 80, 120, 160, 200], fontsize=fsize);
            plt.xlabel('Timesteps', fontsize=fsize2);
            plt.ylabel('MC Index', fontsize=fsize2); 
        else:
            plt.xticks([]);
            
        
        #Similarity plot
        plt.subplot(5, 4, 2*i+2);
        bar_y = []; 
        gammaID = odorID*nTestPerOdor*nGamma;# + sampleNumbers[odorID]*nGamma; 
        for j in range(0, nGamma): 
            bar_y.append(sMatrix[gammaID+j][odorID]); 
            
        #formatting
        plt.bar([40, 80, 120, 160, 200], bar_y, color='black', width=10.0);
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fsize);
        plt.ylim([0.0, 1.02])
        plt.xlim([0, 240])
        if(i>=8): 
            plt.xticks([40, 80, 120, 160, 200], [1, 2, 3, 4, 5], fontsize=fsize);
            plt.xlabel('Gamma Cycles', fontsize=fsize2);
            plt.ylabel('Similarity', fontsize=fsize2); 
        else:
            plt.xticks([]);
    plt.show() 