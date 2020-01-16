# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 02:14:57 2018

@author: ehsan
"""

# -*- coding: utf-8 -*-
"""
@Team: MTL551

"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.transform import resize

def PreProcessing_SingleImage_1(singleImg, filteredImgSize, binarizingTH):     
    filteredImgCenter = int(filteredImgSize/2)   #Center of the Filtered Image (25 here)
    originalImg = singleImg
    originalImg[originalImg > 255] = 255
    binarizedImg = originalImg > binarizingTH  #this needs to be tunned
    origImgSegments = measure.label(binarizedImg, background = 0)
    mostCommonLabel = Counter(origImgSegments.flatten()).most_common(3)
    filterMask = (origImgSegments == mostCommonLabel[1][0]) #+ (origImgSegments ==mostCommonLabel[2][0])
    filteredImg = filterMask * originalImg
    regionImg = measure.regionprops(filterMask.astype(int))[0]
    originalImgCenter = [int(regionImg.bbox[0]+((regionImg.bbox[2]-regionImg.bbox[0])/2)), int(regionImg.bbox[1]+((regionImg.bbox[3]-regionImg.bbox[1])/2))]
    deltaX = regionImg.bbox[2]-regionImg.bbox[0]
    deltaY = regionImg.bbox[3]-regionImg.bbox[1]
    deltaX = int( min(1.5*deltaX , filteredImgSize))
    deltaX += deltaX%2
    deltaY = int( min(1.5*deltaY , filteredImgSize))
    deltaY += deltaY%2
    grabbedImg = []
    grabbedImg = filteredImg[max(0,originalImgCenter[0]-int(deltaX/2)):min(originalImgSize-1,originalImgCenter[0]+int(deltaX/2)), max(0,originalImgCenter[1]-int(deltaY/2)):min(originalImgSize-1,originalImgCenter[1]+int(deltaY/2))]
    tmpXsize = grabbedImg.shape[0]
    tmpYsize = grabbedImg.shape[1]
    tmpImg = np.zeros((filteredImgSize , filteredImgSize))
    tmpImg[filteredImgCenter-int(tmpXsize/2) : filteredImgCenter+int(tmpXsize/2)+tmpXsize%2, filteredImgCenter-int(tmpYsize/2): filteredImgCenter+int(tmpYsize/2)+tmpYsize%2] = grabbedImg
    tmpImgSegments= measure.label(tmpImg>binarizingTH, background = 0)
    mostCommonLabel = Counter(tmpImgSegments.flatten()).most_common()
    finalMask = morphology.remove_small_objects(tmpImg>binarizingTH, min(40, mostCommonLabel[1][1]-1), connectivity=2)
    tmpImg = tmpImg*finalMask
    outImg = tmpImg / np.max(tmpImg)
    return [originalImg, binarizedImg, origImgSegments, filteredImg, grabbedImg, outImg]

    
def PreProcessing_SingleImage_2(sampleImg, filteredImgSize, binarizingTH):
    filteredImgCenter = int(filteredImgSize/2)   #Center of the Filtered Image (25 here)            
    originalImg = sampleImg
    originalImg[originalImg > 255] = 255
    binarizedImg = originalImg > binarizingTH  #this needs to be tunned        
    origImgSegments= measure.label(binarizedImg, background = 0)        
    mostCommonLabel = Counter(origImgSegments.flatten()).most_common(5)
    filterMask = (origImgSegments == mostCommonLabel[1][0]) #+ (origImgSegments ==mostCommonLabel[2][0])
    filteredImg = filterMask * originalImg
    regionImg = measure.regionprops(filterMask.astype(int))[0]
    X1 = regionImg.bbox[0]   #image box cornet point up-left
    X2 = regionImg.bbox[2]   #image box cornet point down-right
    Y1 = regionImg.bbox[1]   #image box cornet point up-left
    Y2 = regionImg.bbox[3]   #image box cornet point down-right
    deltaX = X2 - X1         #vertical - rows
    deltaY = Y2 - Y1         #horizontal - columns
    Xcenter = int(X1+(deltaX/2))  #X-coordinate center of the selected label in the original image
    Ycenter = int(Y1+(deltaY/2))  #Y-coordinate center of the selected label in the original image
    scalingDeltXY = 1
    offsetDeltXY = 3        
    deltaX = (offsetDeltXY+int(scalingDeltXY*deltaX)+int(scalingDeltXY*deltaX)%2)   #Offset
    deltaY = (offsetDeltXY+int(scalingDeltXY*deltaY)+int(scalingDeltXY*deltaY)%2)   #Offset
    X1_grabbing = max(0,Xcenter-int(deltaX/2))                 #up-left corner point in the orignial image
    X2_grabbing = min(originalImgSize-1,Xcenter+int(deltaX/2)) #down-right corner point in the orignial image
    Y1_grabbing = max(0,Ycenter-int(deltaY/2))                 #up-left corner point in the orignial image
    Y2_grabbing = min(originalImgSize-1,Ycenter+int(deltaY/2)) #down-right corner point in the orignial image
    grabbedImg = []    #Grabbed Image from Original Image (not scaled and resized)
    grabbedImg = originalImg[X1_grabbing:X2_grabbing, Y1_grabbing:Y2_grabbing]
    grabbedImgSegments= measure.label(grabbedImg > binarizingTH, background = 0)        
    mostCommonLabel = Counter(grabbedImgSegments.flatten()).most_common(3)
    filteredMaskGrabbed = morphology.remove_small_objects(grabbedImg>binarizingTH, min(45, mostCommonLabel[1][1]-1), connectivity=2)
    grabbedImg = grabbedImg * filteredMaskGrabbed
    filteredGrabbedImg = filteredImg[X1_grabbing:X2_grabbing, Y1_grabbing:Y2_grabbing]
    outDeltaX = np.uint16(deltaX * (filteredImgSize/max(deltaX, deltaY)))  #Scaling
    outDeltaY = np.uint16(deltaY * (filteredImgSize/max(deltaX, deltaY)))  #Scaling
    resizedFiltGrabImg = resize(filteredGrabbedImg, (outDeltaX, outDeltaY))
    tmpImg = np.zeros((filteredImgSize, filteredImgSize))
    tmpImg[filteredImgCenter-int(outDeltaX/2) : filteredImgCenter+int(outDeltaX/2)+outDeltaX%2, filteredImgCenter-int(outDeltaY/2): filteredImgCenter+int(outDeltaY/2)+outDeltaY%2] = resizedFiltGrabImg
    outImg = tmpImg / np.max(tmpImg)
    return [originalImg, binarizedImg, origImgSegments, filteredImg, grabbedImg, outImg]




dataTrain = np.load('train_images.npy', encoding = 'latin1')
dataTest  = np.load('test_images.npy',  encoding = 'latin1')

trainSize = dataTrain.shape[0]               #Number of Training Examples
testSize  = dataTest.shape[0]                #Number of Testing Examples

myCSV = np.genfromtxt('train_labels.csv', delimiter=',', dtype = 'str')
trainLabelWords = myCSV[1:,1]                    #Training labels: Words
uniqueLabelWords = np.unique(trainLabelWords)    #Unique Labels
trainLabel = np.zeros((trainSize,1))             #Training Labels: Numerics  
NumberLabel = uniqueLabelWords.shape[0]
refLabel = np.zeros((NumberLabel,2))
for i in range(NumberLabel):
    trainLabel[trainLabelWords == uniqueLabelWords[i],0] = i


dataSample = dataTrain
#dataSample = dataTest

dataSize = dataSample.shape[0]                #Number of Examples

featureSize = dataSample[0][1].shape[0]       #Original Feature Size which is imageLength*imageWidth
originalImgSize = np.sqrt(featureSize).astype(int)   #Original Image Size

itemList = [654, 351, 345, 9871, 5423, 1263, 7231, 9261, 7384]

fig1 = plt.figure()
fig1.suptitle("Sample of Raw Images", fontsize=11)

cntr = 0
for item in itemList:
    cntr += 1
    sampleImg = dataTrain[item,1].reshape(originalImgSize,originalImgSize)
    sampleLabel = trainLabelWords[item]
    fig1.add_subplot(3, 3, cntr).set_title(sampleLabel, fontsize=9)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(sampleImg,'gray')
plt.show()
fig1.savefig('report/Nine_Sample_Raw_Images.png', dpi=1200)

filteredImgSize = 50
binarizingTH = 5    

itemTitles = ['Original Img', 'Binarized Img', 'Segmented Img', 'Filtered Img', 'Grabbed Img', 'Output Img']
for sampleNumber in range(9):
    sampleImg = dataTrain[itemList[sampleNumber],1].reshape(originalImgSize,originalImgSize)
    sampleLabel = trainLabelWords[itemList[sampleNumber]]
    fig2 = plt.figure()
    fig2.suptitle(sampleLabel, fontsize=11)
    outList = PreProcessing_SingleImage_1(sampleImg, filteredImgSize, binarizingTH)
    cntr = 0
    for item in outList:
        fig2.add_subplot(2, 3, cntr+1).set_title(itemTitles[cntr], fontsize=9)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(item,'gray')
        cntr += 1
    #plt.show()    
    fig2.savefig('report/output_1_sample_'+ str(sampleNumber+1)+ '.png', dpi=1200)


itemTitles = ['Original Img', 'Binarized Img', 'Segmented Img', 'Filtered Img', 'Grabbed Img', 'Output Img']
for sampleNumber in range(9):
    sampleImg = dataTrain[itemList[sampleNumber],1].reshape(originalImgSize,originalImgSize)
    sampleLabel = trainLabelWords[itemList[sampleNumber]]
    fig3 = plt.figure()
    fig3.suptitle(sampleLabel, fontsize=11)
    outList = PreProcessing_SingleImage_2(sampleImg, filteredImgSize, binarizingTH)
    cntr = 0
    for item in outList:
        fig3.add_subplot(2, 3, cntr+1).set_title(itemTitles[cntr], fontsize=9)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(item,'gray')
        cntr += 1
    #plt.show()    
    fig3.savefig('report/output_2_sample_'+ str(sampleNumber+1)+ '.png', dpi=1200)

for sampleNumber in range(9):
    if sampleNumber%3 == 0:
        fig1.savefig('report/report_'+ str(sampleNumber+1)+ '.png', dpi=600)
        cntr = 0
        fig1 = plt.figure()
    sampleImg = dataTrain[itemList[sampleNumber],1].reshape(originalImgSize,originalImgSize)
    sampleLabel = trainLabelWords[itemList[sampleNumber]]
    outList1 = PreProcessing_SingleImage_1(sampleImg, filteredImgSize, binarizingTH)
    outList2 = PreProcessing_SingleImage_2(sampleImg, filteredImgSize, binarizingTH)
    outImg1 = outList1[-1]
    outImg2 = outList2[-1]
    
    if sampleNumber < 3:
        fig1.add_subplot(3, 3, 3*cntr+1)
        plt.imshow(sampleImg,'gray')
        fig1.add_subplot(3, 3, 3*cntr+2)
        plt.imshow(outImg1,'gray')
        fig1.add_subplot(3, 3, 3*cntr+3)
        plt.imshow(outImg2,'gray')
        cntr +=1
    elif sampleNumber > 5: 
        fig1.add_subplot(3, 3, 3*cntr+1)
        plt.imshow(sampleImg,'gray')
        fig1.add_subplot(3, 3, 3*cntr+2)
        plt.imshow(outImg1,'gray')
        fig1.add_subplot(3, 3, 3*cntr+3)
        plt.imshow(outImg2,'gray')
        cntr +=1
    else:
        fig1.add_subplot(3, 3, 3*cntr+1)
        plt.imshow(sampleImg,'gray')
        fig1.add_subplot(3, 3, 3*cntr+2)
        plt.imshow(outImg1,'gray')
        fig1.add_subplot(3, 3, 3*cntr+3)
        plt.imshow(outImg2,'gray')
        cntr +=1
        
fig1.savefig('report/report_'+ str(sampleNumber+1)+ '.png', dpi=600)
