import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.transform import resize

#import os
#print(os.listdir("./all"))

""" Pre-Processing """
def PreProcessing_1(dataSample, filteredImgSize, binarizingTH):
    dataSize = dataSample.shape[0]               #Number of Examples
    NumPixels = dataSample[0][1].shape[0]       #Original Feature Size which is imageLength*imageWidth
    originalImgSize = np.sqrt(NumPixels).astype(int)   #Original Image Size
    """ Initialization """
    filteredImgCenter = int(filteredImgSize/2)   #Center of the Filtered Image (25 here)
    featureSize = filteredImgSize**2
    processedImageSet = np.zeros((dataSize, featureSize))
    """ First Train Set: to find Vocabs of SIFT, SURF and anyother feature """
    for i in range(dataSize):
        if (i+1)%1000 == 0:
            print(i , ' of data have been processed')
        originalImg = dataSample[i,1].reshape(originalImgSize,originalImgSize)
        originalImg[originalImg > 255] = 255
        binarizedImg = originalImg > binarizingTH  #this needs to be tunned
        origImgSegments= measure.label(binarizedImg, background = 0)
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
        processedImageSet[i,:] = outImg.flatten()
    return processedImageSet

def PreProcessing_2(dataSample, filteredImgSize, binarizingTH):
    dataSize = dataSample.shape[0]               #Number of Examples
    NumPixels = dataSample[0][1].shape[0]       #Original Feature Size which is imageLength*imageWidth
    originalImgSize = np.sqrt(NumPixels).astype(int)   #Original Image Size
    """ Initialization """
    featureSize = filteredImgSize**2
    processedImageSet = np.zeros((dataSize, featureSize))
    filteredImgCenter = int(filteredImgSize/2)   #Center of the Filtered Image (25 here)    
    for i in range(dataSize):
        if (i+1)%1000 == 0:
            print(i , ' of data have been processed')
        originalImg = dataSample[i,1].reshape(originalImgSize,originalImgSize)  
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
        filteredMaskGrabbed = morphology.remove_small_objects(grabbedImg>binarizingTH, min(40, mostCommonLabel[1][1]-1), connectivity=2)
        grabbedImg = grabbedImg * filteredMaskGrabbed        
        filteredGrabbedImg = filteredImg[X1_grabbing:X2_grabbing, Y1_grabbing:Y2_grabbing]        
        outDeltaX = np.uint16(deltaX * (filteredImgSize/max(deltaX, deltaY)))  #Scaling
        outDeltaY = np.uint16(deltaY * (filteredImgSize/max(deltaX, deltaY)))  #Scaling        
        resizedFiltGrabImg = resize(filteredGrabbedImg, (outDeltaX, outDeltaY))        
        tmpImg = np.zeros((filteredImgSize, filteredImgSize))
        tmpImg[filteredImgCenter-int(outDeltaX/2) : filteredImgCenter+int(outDeltaX/2)+outDeltaX%2, filteredImgCenter-int(outDeltaY/2): filteredImgCenter+int(outDeltaY/2)+outDeltaY%2] = resizedFiltGrabImg        
        outImg = tmpImg / np.max(tmpImg)        
        processedImageSet[i,:] = outImg.flatten()# image with only one connected object
    return processedImageSet


"""                    Loading Train and Test Data:                         """     
dataTrain = np.load('train_images.npy', encoding = 'latin1')
dataTest  = np.load('test_images.npy',  encoding = 'latin1')
#dataTrain = np.load('./all/train_images.npy', encoding = 'latin1')
#dataTest  = np.load('./all/test_images.npy',  encoding = 'latin1')

"""  Extracting Train & Test Sets Sizes and Original Feature & Image Sizes  """
trainSize = dataTrain.shape[0]               #Number of Training Examples
testSize  = dataTest.shape[0]                #Number of Testing Examples

"""    Loading Labels of Train Data & Converting Words to Numeric Labels    """
myCSV = np.genfromtxt('train_labels.csv', delimiter=',', dtype = 'str')
#myCSV = np.genfromtxt('./all/train_labels.csv', delimiter=',', dtype = 'str')
trainLabelWords = myCSV[1:,1]                    #Training labels: Words
uniqueLabelWords = np.unique(trainLabelWords)    #Unique Labels
trainLabel = np.zeros((trainSize,1))             #Training Labels: Numerics  
NumberLabel = uniqueLabelWords.shape[0]
refLabel = np.zeros((NumberLabel,2))
for i in range(NumberLabel):
    trainLabel[trainLabelWords == uniqueLabelWords[i],0] = i


""" Parameter Initialization """ 
filteredImgSize = 50
binarizingTH = 8     # for first segmentation

""" Pre-Processing Stage: """
print('Pre-Processing Train Data-First Method:')
processedImgTrain1 = PreProcessing_1(dataTrain, filteredImgSize, binarizingTH)
print('Pre-Processing Train Data-Second Method:')
processedImgTrain2 = PreProcessing_2(dataTrain, filteredImgSize, binarizingTH)

print('Pre-Processing Test Data-First Method:')
processedImgTest1  = PreProcessing_1(dataTest,  filteredImgSize, binarizingTH) 
print('Pre-Processing Test Data-Second Method:')
processedImgTest2  = PreProcessing_2(dataTest,  filteredImgSize, binarizingTH)  


nTrain = int(0.9*trainSize)
idxTrainValid = np.random.choice(trainSize, [trainSize,1],replace = False)


np.save('PreProcessing_Method1',[processedImgTrain1, trainLabel, processedImgTest1, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain])
np.save('PreProcessing_Method2',[processedImgTrain2, trainLabel, processedImgTest2, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain])