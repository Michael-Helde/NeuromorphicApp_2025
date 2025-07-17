from lib import epl 
from lib import OSN
from lib import readout
from lib import plots
import numpy as np
import pickle 
import brian2 as b2
import brian2hears as b2h
import matplotlib.pyplot as plt

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
            
            
def my_plotFigure4b(sMatrix, labels):
    bar1_x, bar2_x, bar3_x, bar4_x, bar5_x = [], [], [], [], [];      
    bar1_y, bar2_y, bar3_y, bar4_y, bar5_y = [], [], [], [], [];      
    nGamma = 5; 
    testOdorID = 0;              #sniff ID of test odor 
    for i in range(0, nGamma):
        bar1_y.append(sMatrix[testOdorID+i][0]);
        bar2_y.append(sMatrix[testOdorID+i][4]);
        bar3_y.append(sMatrix[testOdorID+i][5]);
        bar4_y.append(sMatrix[testOdorID+i][1]);
        bar5_y.append(sMatrix[testOdorID+i][2]);
    w = 0.15        #width of bars
    fsize = 14;
    xticks = []
    xtick_labels = []
    for i in range(0, 5):    
        bar1_x.append(i-2*w);
        bar2_x.append(i-w);
        bar3_x.append(i);
        bar4_x.append(i+w); 
        bar5_x.append(i+2*w);     
        xticks.append(i)
        xtick_labels.append(str(i+1))
    plt.figure(2, figsize=(10, 5))
    fig = plt.subplot(111)
    opacity = 0.5; 
    bar1 = plt.bar(bar1_x, bar1_y, width = w, color = 'orange', alpha = opacity, align='center')
    bar2 = plt.bar(bar2_x, bar2_y, width = w, color = 'purple', alpha = opacity, align='center')
    bar3 = plt.bar(bar3_x, bar3_y, width = w, color = 'red', alpha = opacity, align='center')
    bar4 = plt.bar(bar4_x, bar4_y, width = w, color = 'blue', alpha=opacity, align='center')
    bar5 = plt.bar(bar5_x, bar5_y, width = w, color = 'green', alpha=opacity, align='center')
    plt.xlim(-1, len(bar1_x)+0.2)
    plt.ylim(0, 1.02)
    plt.xticks(xticks, xtick_labels, fontsize=fsize)
    plt.yticks(fontsize=fsize); 
    fig.legend( (bar1, bar2, bar3, bar4, bar5), labels, loc = 'upper left', fontsize=fsize)
    plt.ylabel('Similarity', fontsize=fsize); 
    plt.xlabel('Gamma Cycles', fontsize=fsize); 
    plt.title("Similarity of five sounds over five gamma cycles")
    plt.show()
    
    
def learningBarFigure(sMatrix, plotTitles, testOdorID = 0, nGamma = 5): 
    nOdors = len(sMatrix[0]) #for future so that number of bars displayed could be a parameter
    nTestPerOdor = len(sMatrix)/(nOdors*nGamma);       # = 120 / (6*5) = 4  fot     

    bar1_x, bar2_x, bar3_x, bar4_x, bar5_x, bar6_x = [], [], [], [], [], [];      
    bar1_y, bar2_y, bar3_y, bar4_y, bar5_y, bar6_y = [], [], [], [], [], [];      
    
    # odorID = int((testOdorID*nTestPerOdor)*nGamma) # (0*5 * 5) = 0
    odorID = int(testOdorID*(nTestPerOdor+nGamma)) # (0*5 * 5) = 0
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

    
def rasterSimilarityPlot(gammaCode, sMatrix, plotTitles, nMCs=72, nGamma=10, nRows=5, nCols=4):
    """
    Plots paired raster plots and similarity bar plots for each odor. My version of plotFigure4d
    
    Parameters:
    - gammaCode: 2D list [nTotalGamma][nMCs], spike timing values
    - sMatrix: 2D list [nTotalTests][nOdors], similarity scores
    - plotTitles: list of odor names in display order
    - nMCs: number of mitral cells (y-axis for raster)
    - nGamma: number of gamma cycles per odor
    - nRows, nCols: grid layout for subplots
    """
    nOdors = len(sMatrix[0])
    nTestPerOdor = int(len(sMatrix) / (nOdors * nGamma))
    
    fsize = 12
    fsize2 = 16
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Figure 4d – Raster + Similarity per Odor", fontsize=18)
    plt.subplots_adjust(hspace=0.45, wspace=0.25)

    for i in range(nOdors):
        odorID = i
        sniffID = int(nTestPerOdor * odorID)  # First test sniff for this odor
        
        # === Raster Plot ===
        ax_raster = plt.subplot(nRows, nCols, 2 * i + 1)
        for j in range(nGamma):  # gamma cycles
            gammaIdx = int(nGamma * nOdors * 2 + nGamma * sniffID + j)
            for k in range(nMCs):
                if gammaCode[gammaIdx][k] != 0:
                    spikeTime = 20 - gammaCode[gammaIdx][k] + 40 * j
                    ax_raster.scatter(spikeTime, k, color='k', s=2)

        ax_raster.set_title(plotTitles[i], fontsize=fsize2)
        ax_raster.set_xlim([0, 40 * nGamma])
        ax_raster.set_ylim([0, nMCs])
        ax_raster.set_yticks(range(0, nMCs + 1, int(nMCs / 4)))
        ax_raster.tick_params(axis='y', labelsize=fsize)

        if i >= (nRows - 1) * (nCols // 2):
            ax_raster.set_xticks([40 * j for j in range(nGamma)])
            ax_raster.set_xlabel('Timesteps', fontsize=fsize2)
            ax_raster.set_ylabel('MC Index', fontsize=fsize2)
        else:
            ax_raster.set_xticks([])
        
        # === Similarity Bar Plot ===
        ax_bar = plt.subplot(nRows, nCols, 2 * i + 2)
        gammaStartIdx = int(odorID * nTestPerOdor * nGamma)
        bar_y = [sMatrix[gammaStartIdx + j][odorID] for j in range(nGamma)]
        xvals = [40 + 40 * j for j in range(nGamma)]

        ax_bar.bar(xvals, bar_y, color='black', width=10.0)
        ax_bar.set_ylim([0.0, 1.02])
        ax_bar.set_xlim([0, xvals[-1] + 40])
        ax_bar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_bar.tick_params(axis='y', labelsize=fsize)

        if i >= (nRows - 1) * (nCols // 2):
            ax_bar.set_xticks(xvals)
            ax_bar.set_xticklabels([str(j + 1) for j in range(nGamma)], fontsize=fsize)
            ax_bar.set_xlabel('Gamma Cycles', fontsize=fsize2)
            ax_bar.set_ylabel('Similarity', fontsize=fsize2)
        else:
            ax_bar.set_xticks([])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

